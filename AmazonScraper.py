"""
Amazon scraper — Playwright stealth + persistent profile
Outputs unified CSV schema that matches the (patched) Temu scraper.
Schema: product_id,source,title,price,image_url,product_url
Now with checkpoint saving after each page!
"""
import asyncio
import csv
import random
import re
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

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

FIELDNAMES = [
    "product_id",
    "source",
    "title",
    "price",
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
                "source": r["source"],
                "title": r["title"].strip()[:200],
                "price": f"{float(r['price']):.2f}",
                "image_url": r.get("image_url", ""),
                "product_url": r["product_url"],
            })

    print(f"📄 CSV updated → {csv_file} (added {len(rows)} new products)")


def dl_image(row: Dict, dest: Path) -> None:
    dest.mkdir(exist_ok=True)
    file_path = dest / f"{row['product_id']}.jpg"

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
        print(f"✗ {row['product_id']} {exc}")


async def download_images_batch(products: List[Dict], out_dir: Path, desc: str = ""):
    """Download images for a batch of products"""
    pics = [p for p in products if p["image_url"]]
    if not pics:
        return

    print(f"\n🖼️  Downloading images{desc}...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        fut = [pool.submit(dl_image, p, out_dir) for p in pics]
        for _ in tqdm(as_completed(fut), total=len(fut), unit="img"):
            pass
    print(f"✅ Downloaded images for {len(pics)} products{desc}")


async def smooth_scroll(page, rounds: int = 15):
    last_height = await page.evaluate("document.body.scrollHeight")
    for i in range(rounds):
        await page.mouse.wheel(0, 900)
        await asyncio.sleep(random.uniform(1.5, 2.5))
        new_height = await page.evaluate("document.body.scrollHeight")
        if new_height == last_height:
            await page.evaluate("window.scrollBy(0, -500)")
            await asyncio.sleep(0.5)
            await page.mouse.wheel(0, 1000)
            await asyncio.sleep(1)
            final_height = await page.evaluate("document.body.scrollHeight")
            if final_height == new_height:
                break
        last_height = new_height
        if (i + 1) % 5 == 0:
            print(f"  Scrolled {i + 1} times…")


async def wait_for_images(page):
    print("Waiting for images to load…")
    await page.wait_for_function(
        """
        () => {
            const imgs = [...document.querySelectorAll('img')];
            return imgs.filter(img => img.src && /amazon\.com/.test(img.src)).length > 5;
        }
        """,
        timeout=30_000,
    )
    await asyncio.sleep(2)


def extract_asin(url: str) -> str:
    for pat in (r"/dp/([A-Z0-9]{10})", r"/gp/product/([A-Z0-9]{10})", r"[?&]ASIN=([A-Z0-9]{10})"):
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return ""


async def extract(page) -> List[Dict]:
    await wait_for_images(page)
    products, seen = [], set()
    containers = await page.query_selector_all(
        '[data-component-type="s-search-result"], [data-asin]:not([data-asin=""])')
    print(f"Found {len(containers)} potential containers")

    for c in containers:
        try:
            asin = await c.get_attribute("data-asin") or ""
            if not asin or asin in seen:
                continue
            seen.add(asin)
            link = await c.query_selector("h2 a, a[href*='/dp/']")
            if not link:
                continue
            href = await link.get_attribute("href") or ""
            if "aax-us-iad.amazon.com" in href:
                asin = extract_asin(href) or asin
                href = f"{BASE_URL}/dp/{asin}"
            full_url = href if href.startswith("http") else BASE_URL + href
            title_el = await c.query_selector("h2 span, h2 a span")
            title = (await title_el.inner_text()) if title_el else (await link.inner_text())
            title = (title or "").strip()

            # image
            img_url = ""
            for sel in ["img.s-image", "img[src*='m.media-amazon.com']", "img[data-lazy-src]"]:
                img = await c.query_selector(sel)
                if img:
                    img_url = await img.get_attribute("src") or await img.get_attribute("data-lazy-src") or ""
                    if img_url:
                        break

            # price
            price = 0.0
            span = await c.query_selector("span.a-offscreen")
            if span:
                m = re.search(r"[\d,.]+", (await span.inner_text()))
                if m:
                    price = float(m.group().replace(",", ""))

            if not title:
                continue
            products.append({
                "product_id": asin,
                "source": "amazon",
                "title": title,
                "price": price,
                "image_url": img_url,
                "product_url": full_url,
            })
        except Exception:
            continue
    return products


async def handle_captcha(page):
    try:
        cap = await page.query_selector("form[action*='validateCaptcha'], #captchacharacters, .a-box-inner h4")
        if cap:
            print("⚠️  CAPTCHA detected — solve manually.")
            await page.wait_for_selector('[data-component-type="s-search-result"]', timeout=300_000)
    except Exception:
        pass


async def extract_and_save_checkpoint(page, page_number: int = 1):
    """Extract products and save them immediately"""
    print(f"\nExtracting products from page {page_number}...")
    products = await extract(page)
    print(f"Extracted {len(products)} products from page {page_number}")

    # Filter for new products only
    existing_ids = load_existing_ids(CSV_FILE)
    new_products = [p for p in products if p["product_id"] not in existing_ids]

    if new_products:
        save_to_csv(new_products, CSV_FILE)

        # Download images for new products
        await download_images_batch(new_products, OUT_DIR, f" for page {page_number}")
    else:
        print(f"No new products found on page {page_number}")

    return new_products


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Amazon scraper starting…\n" + "=" * 60)

    max_pages = int(input("Enter the number of pages to scrape (default 3): ") or 3)

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
        )
        page = await context.new_page()
        await page.route("**/*", lambda r: r.continue_())
        print(f"Navigating to {SEARCH_URL}")
        await page.goto(SEARCH_URL, wait_until="domcontentloaded")
        await handle_captcha(page)
        try:
            await page.wait_for_selector('[data-component-type="s-search-result"]', timeout=60_000)
        except PWTimeout:
            print("Timeout: results didn't load.")
            return
        await asyncio.sleep(3)
        await smooth_scroll(page)

        all_new_products = []
        page_num = 1

        # Extract and save first page immediately
        first_page_products = await extract_and_save_checkpoint(page, page_number=page_num)
        all_new_products.extend(first_page_products)

        # Continue with remaining pages
        while page_num < max_pages:
            next_btn = await page.query_selector('a.s-pagination-next')
            if not next_btn:
                print("No more pages available")
                break

            page_num += 1
            print(f"\n{'=' * 60}")
            print(f"Navigating to page {page_num}...")

            try:
                async with page.expect_navigation(wait_until="domcontentloaded", timeout=60_000):
                    await next_btn.click()
            except PWTimeout:
                print(f"Timeout navigating to page {page_num}")
                break

            await handle_captcha(page)
            await asyncio.sleep(random.uniform(2, 4))
            await smooth_scroll(page)

            # Extract and save this page immediately
            page_products = await extract_and_save_checkpoint(page, page_number=page_num)
            all_new_products.extend(page_products)

        print(f"\n✓ Total scraped: {len(all_new_products)} new products across {page_num} pages")
        print(f"✓ All data saved to: {CSV_FILE}")
        print(f"✓ Images saved to: {OUT_DIR}/")

        print("\nDone — close the browser to finish.")
        await context.close()


if __name__ == "__main__":
    asyncio.run(main())
