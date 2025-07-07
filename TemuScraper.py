"""
Temu scraper — Playwright stealth + persistent profile
Outputs **exactly** the same unified CSV schema as the Amazon scraper:
product_id,source,title,price,image_url,product_url
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

def save_to_csv(rows: List[Dict], csv_file: Path):
    if not rows:
        print("No products to save")
        return
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
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
    # sanity check
    with csv_file.open("r", encoding="utf-8") as f:
        assert f.readline().strip() == ",".join(FIELDNAMES), "CSV header mismatch"
    print(f"📄 CSV saved → {csv_file}")


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


async def smooth_scroll(page, rounds: int = 25):
    last_height = await page.evaluate("document.body.scrollHeight")
    for i in range(rounds):
        await page.mouse.wheel(0, 900)
        await asyncio.sleep(random.uniform(1.2, 2.4))
        new_height = await page.evaluate("document.body.scrollHeight")
        if new_height == last_height:
            await page.evaluate("window.scrollBy(0, -500)")
            await asyncio.sleep(0.5)
            await page.mouse.wheel(0, 1000)
            await asyncio.sleep(1)
            if await page.evaluate("document.body.scrollHeight") == new_height:
                break
        last_height = new_height
        if (i + 1) % 5 == 0:
            print(f"  Scrolled {i + 1} times…")


async def wait_for_images(page):
    print("Waiting for images…")
    await page.wait_for_function(
        """() => [...document.querySelectorAll('img')]
              .filter(i => i.src && /temu\.com|kwcdn\.com/.test(i.src)).length > 5""",
        timeout=30_000,
    )
    await asyncio.sleep(2)


async def extract(page):
    await wait_for_images(page)
    products, seen = [], set()
    containers = await page.query_selector_all("div[data-tooltip-title], div[class*='goods-item'], div[class*='product-item']")
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
            for sel in ["img", "img[src*='kwcdn.com']"]:
                img = await c.query_selector(sel)
                if img:
                    img_url = await img.get_attribute("src") or await img.get_attribute("data-src") or ""
                    if img_url:
                        break
            # title
            title = (await link.inner_text()).strip()
            if len(title) < 5:
                t_el = await c.query_selector("[class*='title'], [class*='name'], h2, h3")
                title = (await t_el.inner_text()).strip() if t_el else ""
            title = clean_title(title) or f"Temu Baby Toy {pid}"
            # price
            price = 0.0
            cand = [
                "div[data-type='price'][aria-label]",
                "[data-price]",
                "[class*='price']",
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

    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            PROFILE_DIR,
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
            viewport={"width": 1280, "height": 780},
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
        await smooth_scroll(page)
        print("Extracting…")
        products = await extract(page)
        print(f"✓ Scraped {len(products)} products")
        if products:
            save_to_csv(products, CSV_FILE)
            print("\n🖼️  Downloading images…")
            from concurrent.futures import ThreadPoolExecutor, as_completed
            pics = [p for p in products if p["image_url"]]
            with ThreadPoolExecutor(max_workers=8) as pool:
                fut = [pool.submit(dl_image, p, OUT_DIR) for p in pics]
                for _ in tqdm(as_completed(fut), total=len(fut), unit="img"):
                    pass
            print(f"✅ Images saved to {OUT_DIR}")
        else:
            print("❌ No products extracted — selectors may need update.")
        print("\nClose the browser to finish.")
        await ctx.close()


if __name__ == "__main__":
    asyncio.run(main())