"""
Amazon scraper — Playwright stealth + persistent profile
Based on the Temu scraper structure, adapted for Amazon
"""
import asyncio
import csv
import random
import re
from pathlib import Path
from typing import Dict, List
import requests
from tqdm import tqdm
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# ── Config ────────────────────────────────────────────────────────────────────
SEARCH_URL = "https://www.amazon.com/s?k=baby+toys&ref=nb_sb_noss"
BASE_URL = "https://www.amazon.com"
PROFILE_DIR = Path("amazon_profile")
OUT_DIR = Path("amazon_baby_toys_imgs")
CSV_FILE = Path("amazon_baby_toys.csv")
UA_STRING = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def dl_image(row: Dict, dest: Path) -> None:
    dest.mkdir(exist_ok=True)
    file_path = dest / f"{row['amazon_id']}.jpg"

    if file_path.exists() or not row["image_url"]:
        return

    headers = {
        "User-Agent": UA_STRING,
        "Referer": "https://www.amazon.com/",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        r = requests.get(row["image_url"], headers=headers, timeout=15)
        r.raise_for_status()
        file_path.write_bytes(r.content)
    except Exception as exc:
        print(f"✗ {row['amazon_id']} {exc}")


async def smooth_scroll(page, rounds: int = 15):
    """Smooth scrolling to load all products"""
    last_height = await page.evaluate("document.body.scrollHeight")

    for i in range(rounds):
        await page.mouse.wheel(0, 900)
        await asyncio.sleep(random.uniform(1.5, 2.5))

        new_height = await page.evaluate("document.body.scrollHeight")
        if new_height == last_height:
            # Try once more with a different scroll
            await page.evaluate("window.scrollBy(0, -500)")
            await asyncio.sleep(0.5)
            await page.mouse.wheel(0, 1000)
            await asyncio.sleep(1)

            final_height = await page.evaluate("document.body.scrollHeight")
            if final_height == new_height:
                break

        last_height = new_height

        # Print progress
        if (i + 1) % 5 == 0:
            print(f"  Scrolled {i + 1} times...")


async def wait_for_images(page):
    """Wait for images to load"""
    print("Waiting for images to load...")

    # Wait for Amazon product images
    await page.wait_for_function(
        """
        () => {
            const images = document.querySelectorAll('img');
            const loadedImages = Array.from(images).filter(img => 
                img.src && 
                img.src.startsWith('http') && 
                (img.src.includes('images-na.ssl-images-amazon.com') || 
                 img.src.includes('m.media-amazon.com'))
            );
            return loadedImages.length > 5;
        }
        """,
        timeout=30000
    )

    # Additional wait to ensure images are fully loaded
    await asyncio.sleep(2)


async def extract_asin(url: str) -> str:
    """Extract ASIN from Amazon product URL"""
    # Pattern 1: /dp/ASIN
    m = re.search(r"/dp/([A-Z0-9]{10})", url)
    if m:
        return m.group(1)

    # Pattern 2: /gp/product/ASIN
    m = re.search(r"/gp/product/([A-Z0-9]{10})", url)
    if m:
        return m.group(1)

    # Pattern 3: ASIN in query params
    m = re.search(r"[?&]ASIN=([A-Z0-9]{10})", url)
    if m:
        return m.group(1)

    return ""


async def extract(page) -> List[Dict]:
    """Extract product information from Amazon search results"""
    # First wait for images to load
    await wait_for_images(page)

    # Get all product containers
    products = []
    seen = set()

    # Amazon uses data-component-type="s-search-result" for products
    product_containers = await page.query_selector_all(
        '[data-component-type="s-search-result"], [data-asin]:not([data-asin=""])'
    )

    print(f"Found {len(product_containers)} potential product containers")

    for container in product_containers:
        try:
            # Get ASIN directly from data attribute
            asin = await container.get_attribute("data-asin")
            if not asin or asin in seen:
                continue

            seen.add(asin)

            # Find the main product link
            link = await container.query_selector("h2 a, .s-link, a[href*='/dp/']")
            if not link:
                continue

            href = await link.get_attribute("href")
            if not href:
                continue

            # Check if the URL is a tracking URL and clean it up
            if "aax-us-iad.amazon.com" in href:
                # Extract the ASIN from the URL
                asin = await extract_asin(href)
                if asin:
                    href = f"{BASE_URL}/dp/{asin}"

            # Now href should be the clean product URL
            full_url = href if href.startswith("http") else BASE_URL + href

            # Get title
            title_text = ""
            title_el = await container.query_selector("h2 span, h2 a span, .s-size-mini-space-unit span")
            if title_el:
                title_text = (await title_el.inner_text() or "").strip()

            if not title_text:
                title_text = (await link.inner_text() or "").strip()

            # Find image with multiple strategies
            img_url = ""

            # Strategy 1: Direct img tag with class s-image
            img_el = await container.query_selector("img.s-image")
            if img_el:
                img_url = await img_el.get_attribute("src")

            # Strategy 2: Any img tag with Amazon CDN
            if not img_url:
                img_el = await container.query_selector(
                    "img[src*='images-na.ssl-images-amazon.com'], img[src*='m.media-amazon.com']")
                if img_el:
                    img_url = await img_el.get_attribute("src")

            # Strategy 3: Check data-lazy-src for lazy loading
            if not img_url:
                img_el = await container.query_selector("img[data-lazy-src]")
                if img_el:
                    img_url = await img_el.get_attribute("data-lazy-src")

            # Strategy 4: Get any image in the container
            if not img_url:
                all_imgs = await container.query_selector_all("img")
                for img in all_imgs:
                    src = await img.get_attribute("src")
                    if src and src.startswith("http") and "transparent-pixel" not in src:
                        img_url = src
                        break

            # ── Price ────────────────────────────────────────────────
            price = 0.0

            # 1. Look for the accessible price span that always contains the full value
            price_span = await container.query_selector("span.a-offscreen")
            if price_span:
                price_text = (await price_span.inner_text() or "").strip()

                # Example formats: "$9.99", "$17.99 Save 10%", "$6.99 - $12.99"
                m = re.search(r"\$?\s*([\d]+(?:[\d,]*)(?:\.\d+)?)", price_text)
                if m:
                    price = float(m.group(1).replace(",", ""))

            # 2. Fallbacks – keep your old selectors just in case
            if price == 0.0:
                fallback_selectors = [
                    "span.a-price-whole",
                    "[data-a-color='base'] span.a-offscreen",
                    ".a-price-range span.a-offscreen",
                ]
                for sel in fallback_selectors:
                    el = await container.query_selector(sel)
                    if not el:
                        continue
                    txt = (await el.inner_text() or "").strip()
                    m = re.search(r"\$?\s*([\d]+(?:[\d,]*)(?:\.\d+)?)", txt)
                    if m:
                        price = float(m.group(1).replace(",", ""))
                        break

            # 3. Last-ditch: search the container’s HTML
            if price == 0.0:
                html_snippet = await container.inner_html()
                m = re.search(r"\$([\d]+(?:[\d,]*)(?:\.\d+)?)", html_snippet)
                if m:
                    price = float(m.group(1).replace(",", ""))

                    products.append({
                        "amazon_id": asin,
                        "title": title_text[:200],
                        "price": price,  # <-- this is where the value is used
                        "image_url": img_url,
                        "product_url": full_url,
                    })
        except Exception as e:
            continue

    return products


async def handle_captcha(page):
    """Check for and handle CAPTCHA if present"""
    try:
        # Check for common CAPTCHA indicators
        captcha_present = await page.query_selector(
            "form[action*='validateCaptcha'], #captchacharacters, .a-box-inner h4")
        if captcha_present:
            print("⚠️  CAPTCHA detected! Please solve it manually in the browser.")
            print("   Waiting for you to solve the CAPTCHA...")

            # Wait for CAPTCHA to be solved (wait for search results)
            await page.wait_for_selector(
                '[data-component-type="s-search-result"]',
                timeout=300000  # 5 minutes
            )
            print("✓ CAPTCHA solved, continuing...")
            await asyncio.sleep(2)
    except:
        pass


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Amazon Scraper - Starting...")
    print("=" * 60)

    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(
            PROFILE_DIR,
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
            viewport={"width": 1280, "height": 780},
            user_agent=UA_STRING,
            locale="en-US",
            permissions=[],
            geolocation=None,
        )

        # Add stealth scripts
        await context.add_init_script("""
            // Override the navigator.webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            
            // Fix chrome object
            window.chrome = {
                runtime: {},
            };
            
            // Fix permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        """)

        page = await context.new_page()

        # Enable image loading
        await page.route("**/*", lambda route: route.continue_())

        print(f"Navigating to: {SEARCH_URL}")
        await page.goto(SEARCH_URL, wait_until="domcontentloaded")

        # Handle potential CAPTCHA
        await handle_captcha(page)

        # Wait for initial content
        try:
            await page.wait_for_selector(
                '[data-component-type="s-search-result"], [data-asin]',
                timeout=60000,
            )
        except PWTimeout:
            print("Timeout: products did not load. Amazon might be blocking or showing CAPTCHA.")
            await context.close()
            return

        # Additional wait for dynamic content
        await asyncio.sleep(3)

        print("Scrolling to load more products...")
        await smooth_scroll(page)

        # Check if there's a "Next" button to load more results
        all_products = []
        page_num = 1
        max_pages = 3  # Limit to prevent too long scraping

        while page_num <= max_pages:
            print(f"\nExtracting products from page {page_num}...")
            products = await extract(page)
            all_products.extend(products)
            print(f"✓ Found {len(products)} products on page {page_num}")

            next_button = await page.query_selector('a.s-pagination-next')
            if not next_button or page_num >= max_pages:
                break  # no more pages or hit max_pages

            print(f"Going to page {page_num + 1}…")

            try:
                # Many times Amazon triggers a full navigation
                async with page.expect_navigation(wait_until="domcontentloaded", timeout=60_000):
                    await next_button.click()
            except PWTimeout:
                # If Amazon swapped results via XHR only (no nav event),
                # just wait until new search-result nodes appear.
                await page.wait_for_selector(
                    '[data-component-type="s-search-result"], [data-asin]',
                    timeout=60_000
                )

            await handle_captcha(page)
            await asyncio.sleep(random.uniform(2, 4))
            await smooth_scroll(page)

        # Remove duplicates
        unique_products = []
        seen_ids = set()
        for p in all_products:
            if p['amazon_id'] not in seen_ids:
                seen_ids.add(p['amazon_id'])
                unique_products.append(p)

        print(f"\n✓ Total scraped: {len(unique_products)} unique products")

        # Debug: Print first few products
        if unique_products:
            print("\nSample products:")
            for p in unique_products[:3]:
                print(f"  ASIN: {p['amazon_id']}")
                print(f"  Title: {p['title'][:50]}...")
                print(f"  Price: ${p['price']}")
                print(f"  Image: {p['image_url'][:50]}..." if p['image_url'] else "  Image: No image")
                print(f"  URL: {p['product_url'][:50]}...")
                print()

        if unique_products:
            # Save to CSV
            with CSV_FILE.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=unique_products[0].keys())
                writer.writeheader()
                writer.writerows(unique_products)
            print(f"\n📄 CSV saved → {CSV_FILE}")

            # Download images
            print("\n🖼️  Downloading images...")
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Filter products with valid image URLs
            products_with_images = [p for p in unique_products if p['image_url']]
            print(f"Found {len(products_with_images)} products with images")

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(dl_image, p, OUT_DIR) for p in products_with_images]
                for _ in tqdm(as_completed(futures), total=len(futures), unit="img"):
                    pass

            print(f"✅ Images stored in {OUT_DIR}")
        else:
            print("❌ No products extracted — Amazon might be blocking or selectors need update.")

        print("\n🚪 Close the browser window to end session.")
        await context.close()


if __name__ == "__main__":
    asyncio.run(main())
