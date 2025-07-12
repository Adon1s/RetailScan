"""
Debugged Temu scraper with improved scrolling and button detection
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
SEARCH_URL = "https://www.temu.com/search_result.html?search_key=baby%20toys&search_method=user"
BASE_URL = "https://www.temu.com"
PROFILE_DIR = Path("temu_profile")
OUT_DIR = Path("temu_baby_toys_imgs")
CSV_FILE = Path("temu_baby_toys.csv")
UA_STRING = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Unified CSV header
FIELDNAMES = [
    "product_id",  # numeric Temu id
    "source",      # "temu"
    "title",
    "price",       # two‑decimal string
    "image_url",
    "product_url",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_existing_ids(csv_file: Path) -> set:
    existing_ids = set()
    if csv_file.exists():
        with csv_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row["product_id"])
    return existing_ids

def save_to_csv(rows: List[Dict], csv_file: Path):
    if not rows:
        print("No new products to save")
        return

    file_exists = csv_file.exists()
    with csv_file.open("a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow({
                "product_id": r["product_id"],
                "source":     r["source"],
                "title":      r["title"].strip()[:200],
                "price":      f"{float(r['price']):.2f}",
                "image_url":  r.get("image_url", ""),
                "product_url": r["product_url"],
            })

    print(f"📄 CSV updated → {csv_file} (added {len(rows)} new products)")

def clean_title(text: str) -> str:
    text = re.sub(r"^Local\s*\n?", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).replace("\n", " ").replace("\r", " ")
    return text.strip()[:200]

def dl_image(row: Dict, dest: Path):
    dest.mkdir(exist_ok=True)
    file_path = dest / f"{row['product_id']}.jpg"
    if file_path.exists() or not row["image_url"]:
        return

    try:
        r = requests.get(row["image_url"], headers={"User-Agent": UA_STRING}, timeout=15)
        r.raise_for_status()
        file_path.write_bytes(r.content)
    except Exception as exc:
        print(f"✗ {row['product_id']} {exc}")

async def check_for_popups(page):
    """Close any popups that might be blocking the page"""
    popup_selectors = [
        "button[aria-label*='close' i]",
        "button[class*='close']",
        "div[class*='modal'] button",
        "svg[class*='close']",
        "[role='button'][aria-label*='close' i]"
    ]

    for selector in popup_selectors:
        try:
            buttons = await page.query_selector_all(selector)
            for btn in buttons:
                if await btn.is_visible():
                    await btn.click()
                    print("✓ Closed a popup")
                    await asyncio.sleep(1)
                    return True
        except:
            continue
    return False

async def smooth_scroll_with_debug(page, max_rounds: int = 50):
    """Improved scrolling with better debugging"""
    print("Starting smooth scroll...")
    last_height = await page.evaluate("document.body.scrollHeight")
    last_position = 0
    rounds = 0
    stuck_count = 0

    while rounds < max_rounds:
        # Get current scroll position
        current_position = await page.evaluate("window.pageYOffset")

        # Check if we're stuck
        if abs(current_position - last_position) < 100:
            stuck_count += 1
            print(f"  ⚠️  Scroll might be stuck (count: {stuck_count})")

            if stuck_count > 3:
                # Try different scroll strategies
                print("  Trying alternative scroll methods...")

                # Method 1: Scroll to specific element
                try:
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(1)
                except:
                    pass

                # Method 2: Find last visible product and scroll to it
                try:
                    await page.evaluate("""
                        const products = document.querySelectorAll('div[data-tooltip-title], div[class*="goods-item"]');
                        if (products.length > 0) {
                            products[products.length - 1].scrollIntoView({behavior: 'smooth', block: 'end'});
                        }
                    """)
                    await asyncio.sleep(1)
                except:
                    pass

                stuck_count = 0
        else:
            stuck_count = 0

        last_position = current_position

        # Regular scroll
        scroll_distance = random.randint(800, 1500)
        await page.mouse.wheel(0, scroll_distance)
        await asyncio.sleep(random.uniform(0.5, 1.2))

        # Check for new content
        new_height = await page.evaluate("document.body.scrollHeight")

        if new_height == last_height:
            # Try scrolling up and down to trigger lazy loading
            print("  No new content, trying to trigger lazy loading...")
            await page.evaluate("window.scrollBy(0, -500)")
            await asyncio.sleep(0.5)
            await page.mouse.wheel(0, 2000)
            await asyncio.sleep(1.5)

            final_height = await page.evaluate("document.body.scrollHeight")
            if final_height == new_height:
                print("  Reached end of current content")
                break
        else:
            print(f"  New content loaded: {new_height - last_height} pixels added")

        last_height = new_height
        rounds += 1

        if rounds % 10 == 0:
            print(f"  Scrolled {rounds} times, current height: {new_height}")
            # Check for popups periodically
            await check_for_popups(page)

async def find_and_click_see_more(page):
    """Button detection using Playwright's get_by_role for specificity"""
    print("\nLooking for 'See more' button...")

    try:
        button = page.get_by_role("button", name="See more", exact=True)
        if await button.is_visible():
            await button.scroll_into_view_if_needed(timeout=20000)
            await button.click(timeout=20000)
            print("  ✓ Clicked 'See more' button successfully!")
            return True
        else:
            print("  'See more' button not visible.")
    except PWTimeout:
        print("  Timeout finding or clicking 'See more' button.")
    except Exception as e:
        print(f"  Error finding/clicking button: {e}")

    return False

async def load_more_pages(page, num_pages: int):
    """Improved pagination with better error handling"""
    loaded_pages = 1

    while loaded_pages < num_pages:
        print(f"\n{'='*60}")
        print(f"Loading page {loaded_pages + 1} of {num_pages}...")

        # First, scroll to bottom
        await smooth_scroll_with_debug(page, max_rounds=30)

        # Check for popups
        await check_for_popups(page)

        # Count current products
        current_products = len(await page.query_selector_all("div[data-tooltip-title], div[class*='goods-item']"))
        print(f"Current products visible: {current_products}")

        # Try to find and click button
        button_clicked = await find_and_click_see_more(page)

        if not button_clicked:
            print("⚠️  Couldn't find 'See more' button")
            print("Possible reasons:")
            print("  - No more pages available")
            print("  - Button text/structure changed")
            print("  - Button is hidden or disabled")

            # Try manual intervention
            print("\n🔧 Manual intervention needed!")
            print("Please scroll down and click 'See more' if you see it.")
            input("Press Enter after clicking (or press Enter to skip)...")

            # Check if new content loaded
            new_products = len(await page.query_selector_all("div[data-tooltip-title], div[class*='goods-item']"))
            if new_products > current_products:
                print(f"✓ New products loaded: {new_products - current_products} added")
                loaded_pages += 1
                continue
            else:
                print("No new products loaded, stopping pagination")
                break

        # Wait for new content
        print("Waiting for new products to load...")
        try:
            await page.wait_for_function(
                f"document.querySelectorAll('div[data-tooltip-title], div[class*=\"goods-item\"]').length > {current_products}",
                timeout=30000
            )

            # Additional scroll to load lazy images
            await smooth_scroll_with_debug(page, max_rounds=20)

            new_products = len(await page.query_selector_all("div[data-tooltip-title], div[class*='goods-item']"))
            print(f"✓ New products loaded: {new_products - current_products} added")

            loaded_pages += 1

        except PWTimeout:
            print("⚠️  Timeout waiting for new products")
            break

        # Random delay between pages
        delay = random.uniform(3, 5)
        print(f"Waiting {delay:.1f}s before next page...")
        await asyncio.sleep(delay)

    print(f"\n✓ Loaded {loaded_pages} pages total")

async def wait_for_images(page, min_images: int = 20):
    """Wait for images with better error handling"""
    print(f"Waiting for at least {min_images} images...")
    try:
        await page.wait_for_function(
            f"""() => {{
                const imgs = [...document.querySelectorAll('img')];
                const validImgs = imgs.filter(i => i.src && (i.src.includes('temu.com') || i.src.includes('kwcdn.com')));
                console.log('Valid images found:', validImgs.length);
                return validImgs.length >= {min_images};
            }}""",
            timeout=30000,
        )
        print(f"✓ Found at least {min_images} images")
    except PWTimeout:
        print(f"⚠️  Timeout waiting for {min_images} images, continuing anyway")

    await asyncio.sleep(2)

# Keep the extract function and other helpers the same...
async def extract(page):
    # [Same as your original extract function]
    await wait_for_images(page, min_images=50)
    products, seen = [], set()

    containers = await page.query_selector_all("div[data-tooltip-title], div[class*='goods-item'], div[class*='product-item'], div[class*='_1p5bt7k8']")
    print(f"Found {len(containers)} potential containers")

    for c in containers:
        try:
            link = await c.query_selector("a[href*='g-']")
            if not link:
                continue

            href = await link.get_attribute("href") or ""
            m = re.search(r"-g-(\d+)\.html", href)
            if not m:
                continue

            pid = m.group(1)
            if pid in seen:
                continue
            seen.add(pid)

            full_url = href if href.startswith("http") else BASE_URL + href

            # image
            img_url = ""
            for sel in ["img", "img[src*='kwcdn.com']", "img[data-src*='kwcdn.com']"]:
                img = await c.query_selector(sel)
                if img:
                    img_url = await img.get_attribute("src") or await img.get_attribute("data-src") or ""
                    if img_url and not img_url.startswith('data:'):
                        break

            # title
            title = (await link.inner_text()).strip()
            if len(title) < 5:
                t_el = await c.query_selector("[class*='title'], [class*='name'], h2, h3, div[class*='_1f1og7gq']")
                title = (await t_el.inner_text()).strip() if t_el else ""
            title = clean_title(title) or f"Temu Baby Toy {pid}"

            # price
            price = 0.0
            cand = [
                "div[data-type='price'][aria-label]",
                "[data-price]",
                "[class*='price']",
                "div[class*='_17l7u59q']",
            ]
            for sel in cand:
                el = await c.query_selector(sel)
                if not el:
                    continue
                raw = (await el.get_attribute("aria-label")) or (await el.get_attribute("data-price")) or (await el.inner_text())
                m_p = re.search(r"\d+[\d,]*(?:\.\d+)?", raw or "")
                if m_p:
                    price = float(m_p.group().replace(",", ""))
                    break

            if price == 0.0:
                html = await c.inner_html()
                m_p = re.search(r"\d+[\d,]*(?:\.\d+)?", html)
                if m_p:
                    price = float(m_p.group().replace(",", ""))

            products.append({
                "product_id": pid,
                "source": "temu",
                "title": title,
                "price": price,
                "image_url": img_url,
                "product_url": full_url,
            })
        except Exception:
            continue

    return products

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Temu scraper starting…\n" + "=" * 60)

    num_pages = int(input("How many pages of results to scrape? (1 or more): "))

    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            PROFILE_DIR,
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
            viewport={"width": 1366, "height": 768},
            user_agent=UA_STRING,
            locale="en-US",
        )

        page = await ctx.new_page()
        await page.route("**/*", lambda r: r.continue_())

        print(f"Navigating to {SEARCH_URL}")
        await page.goto(SEARCH_URL, wait_until="networkidle")

        try:
            await page.wait_for_selector("img, div[data-tooltip-title]", timeout=120_000)
        except PWTimeout:
            print("Timeout: no products loaded.")
            return

        await asyncio.sleep(3)
        await wait_for_images(page, min_images=5)

        # Load additional pages
        await load_more_pages(page, num_pages)

        # Final scroll to ensure all content is loaded
        print("\nDoing final scroll to load all lazy content...")
        await smooth_scroll_with_debug(page, max_rounds=50)

        print("\nExtracting products...")
        products = await extract(page)
        print(f"✓ Scraped {len(products)} products total")

        # Filter for new products only
        existing_ids = load_existing_ids(CSV_FILE)
        new_products = [p for p in products if p["product_id"] not in existing_ids]

        if new_products:
            save_to_csv(new_products, CSV_FILE)

            print("\n🖼️  Downloading images…")
            from concurrent.futures import ThreadPoolExecutor, as_completed
            pics = [p for p in new_products if p["image_url"]]
            with ThreadPoolExecutor(max_workers=8) as pool:
                fut = [pool.submit(dl_image, p, OUT_DIR) for p in pics]
                for _ in tqdm(as_completed(fut), total=len(fut), unit="img"):
                    pass
            print(f"✅ Images saved to {OUT_DIR}")
        else:
            print("No new products extracted.")

        print("\nClose the browser to finish.")
        await ctx.close()

if __name__ == "__main__":
    asyncio.run(main())
