"""
Temu scraper — Playwright stealth + persistent profile
Fixed to capture image URLs and complete product URLs
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
SEARCH_URL = (
    "https://www.temu.com/search_result.html?search_key=baby%20toys&search_method=user"
)
BASE_URL = "https://www.temu.com"
PROFILE_DIR = Path("temu_profile")
OUT_DIR = Path("temu_baby_toys_imgs")
CSV_FILE = Path("temu_baby_toys.csv")
UA_STRING = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def dl_image(row: Dict, dest: Path) -> None:
    dest.mkdir(exist_ok=True)
    file_path = dest / f"{row['temu_id']}.jpg"

    if file_path.exists() or not row["image_url"]:
        return

    headers = {
        "User-Agent": UA_STRING,
        "Referer": "https://www.temu.com/",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        r = requests.get(row["image_url"], headers=headers, timeout=15)
        r.raise_for_status()
        file_path.write_bytes(r.content)
    except Exception as exc:
        print(f"✗ {row['temu_id']} {exc}")


async def smooth_scroll(page, rounds: int = 25):
    """Smooth scrolling to load all products"""
    last_height = await page.evaluate("document.body.scrollHeight")

    for i in range(rounds):
        await page.mouse.wheel(0, 900)
        await asyncio.sleep(random.uniform(1.2, 2.4))

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

    # Wait for at least some images to have src attributes
    await page.wait_for_function(
        """
        () => {
            const images = document.querySelectorAll('img');
            const loadedImages = Array.from(images).filter(img => 
                img.src && 
                img.src.startsWith('http') && 
                (img.src.includes('images.temu.com') || img.src.includes('img.kwcdn.com'))
            );
            return loadedImages.length > 5;
        }
        """,
        timeout=30000
    )

    # Additional wait to ensure images are fully loaded
    await asyncio.sleep(2)


async def extract(page) -> List[Dict]:
    """Extract product information with proper image and URL handling"""
    # First wait for images to load
    await wait_for_images(page)

    # Get all product containers
    products = []
    seen = set()

    # Try multiple selectors for product containers
    product_containers = await page.query_selector_all(
        "div[data-tooltip-title], div[class*='goods-item'], div[class*='product-item']"
    )

    print(f"Found {len(product_containers)} potential product containers")

    for container in product_containers:
        try:
            # Find the link within the container
            link = await container.query_selector("a[href*='g-']")
            if not link:
                continue

            href = await link.get_attribute("href")
            if not href:
                continue

            # Extract product ID
            m = re.search(r"-g-(\d+)\.html", href)
            if not m:
                continue

            pid = m.group(1)
            if pid in seen:
                continue

            seen.add(pid)

            # Build full URL
            full_url = href if href.startswith("http") else BASE_URL + href

            # Find image with multiple strategies
            img_url = ""

            # Strategy 1: Direct img tag within the link
            img_el = await link.query_selector("img")
            if img_el:
                img_url = await img_el.get_attribute("src")

            # Strategy 2: Look for image in the container
            if not img_url:
                img_el = await container.query_selector("img[src*='images.temu.com'], img[src*='img.kwcdn.com']")
                if img_el:
                    img_url = await img_el.get_attribute("src")

            # Strategy 3: Get the first image with http URL
            if not img_url:
                all_imgs = await container.query_selector_all("img")
                for img in all_imgs:
                    src = await img.get_attribute("src")
                    if src and src.startswith("http"):
                        img_url = src
                        break

            # Strategy 4: Check data-src attribute (lazy loading)
            if not img_url:
                img_el = await container.query_selector("img[data-src]")
                if img_el:
                    img_url = await img_el.get_attribute("data-src")

            # Get title text
            title_text = ""

            # Try to get title from link text
            title_text = (await link.inner_text() or "").strip()

            # If no title, try other selectors
            if not title_text or len(title_text) < 5:
                title_el = await container.query_selector("[class*='title'], [class*='name'], h2, h3")
                if title_el:
                    title_text = (await title_el.inner_text() or "").strip()

            # Default title if still empty
            if not title_text:
                title_text = f"Product {pid}"

            # Get price (if visible)
            price = 0.0
            price_el = await container.query_selector("[class*='price']")
            if price_el:
                price_text = await price_el.inner_text()
                # Extract numeric price
                price_match = re.search(r'[\d,]+\.?\d*', price_text)
                if price_match:
                    price = float(price_match.group().replace(',', ''))

            products.append({
                "temu_id": pid,
                "title": title_text[:200],  # Limit title length
                "price": price,
                "image_url": img_url,
                "product_url": full_url,
            })

        except Exception as e:
            # Continue on error for individual products
            continue

    return products


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Temu Scraper - Starting...")
    print("=" * 60)

    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(
            PROFILE_DIR,
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process"
            ],
            viewport={"width": 1280, "height": 780},
            user_agent=UA_STRING,
            locale="en-US",
        )

        page = await context.new_page()

        # Enable image loading (in case it's disabled)
        await page.route("**/*", lambda route: route.continue_())

        print(f"Navigating to: {SEARCH_URL}")
        await page.goto(SEARCH_URL, wait_until="networkidle")

        # Wait for initial content
        try:
            await page.wait_for_selector(
                "img, div[data-tooltip-title]",
                timeout=120000,
            )
        except PWTimeout:
            print("Timeout: products did not load — complete login and retry.")
            await context.close()
            return

        # Additional wait for dynamic content
        await asyncio.sleep(3)

        print("Scrolling to load all products...")
        await smooth_scroll(page)

        print("Extracting products...")
        products = await extract(page)

        print(f"✓ Scraped {len(products)} products")

        # Debug: Print first few products to check data
        if products:
            print("\nSample products:")
            for p in products[:3]:
                print(f"  ID: {p['temu_id']}")
                print(f"  Title: {p['title'][:50]}...")
                print(f"  Price: ${p['price']}")
                print(f"  Image: {p['image_url'][:50]}..." if p['image_url'] else "  Image: No image")
                print(f"  URL: {p['product_url'][:50]}...")
                print()

        if products:
            # Save to CSV
            with CSV_FILE.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=products[0].keys())
                writer.writeheader()
                writer.writerows(products)
            print(f"📄 CSV saved → {CSV_FILE}")

            # Download images
            print("\n🖼️  Downloading images...")
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Filter products with valid image URLs
            products_with_images = [p for p in products if p['image_url']]
            print(f"Found {len(products_with_images)} products with images")

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(dl_image, p, OUT_DIR) for p in products_with_images]
                for _ in tqdm(as_completed(futures), total=len(futures), unit="img"):
                    pass

            print(f"✅ Images stored in {OUT_DIR}")
        else:
            print("❌ No products extracted — selectors may need update.")

        print("\n🚪 Close the browser window to end session.")
        await context.close()

 
if __name__ == "__main__":
    asyncio.run(main())