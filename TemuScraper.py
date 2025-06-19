#!/usr/bin/env python3
"""
Temu scraper — Playwright stealth + persistent profile
Fixes:
- Corrected indentation in `extract()` (syntax error)
- Consistent 120 000‑ms timeout literal (works fine)
- Minor PEP‑8 clean‑up
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
    last_height = await page.evaluate("document.body.scrollHeight")
    for _ in range(rounds):
        await page.mouse.wheel(0, 900)
        await asyncio.sleep(random.uniform(1.2, 2.4))
        new_height = await page.evaluate("document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


async def extract(page) -> List[Dict]:
    anchors = await page.query_selector_all("div[data-tooltip-title] a[href*='g-']")
    seen, products = set(), []
    for a in anchors:
        href = await a.get_attribute("href")
        if not href:
            continue
        m = re.search(r"-g-(\d+)\.html", href)
        if not m:
            continue
        pid = m.group(1)
        if pid in seen:
            continue
        seen.add(pid)

        img_el = await a.query_selector("img.goods-img-external, img")
        img_url = await img_el.get_attribute("src") if img_el else ""
        title_text = (await a.inner_text() or "Untitled").strip()

        products.append(
            {
                "temu_id": pid,
                "title": title_text,
                "price": 0.0,
                "image_url": img_url,
                "product_url": href,
            }
        )
    return products


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(
            PROFILE_DIR,
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
            viewport={"width": 1280, "height": 780},
            user_agent=UA_STRING,
            locale="en-US",
        )
        page = await context.new_page()
        await page.goto(SEARCH_URL)

        try:
            await page.wait_for_selector(
                "img.goods-img-external, div[data-tooltip-title] img",
                timeout=120000,
            )
        except PWTimeout:
            print("Timeout: products did not load — complete login and retry.")
            await context.close()
            return

        await smooth_scroll(page)
        products = await extract(page)
        print(f"✓ scraped {len(products)} products")

        if products:
            with CSV_FILE.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=products[0].keys())
                writer.writeheader()
                writer.writerows(products)
            print("📄 CSV saved →", CSV_FILE)

            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(dl_image, p, OUT_DIR) for p in products]
                for _ in tqdm(as_completed(futures), total=len(futures), unit="img"):
                    pass
            print("🖼 images stored in", OUT_DIR)
        else:
            print("No products extracted — selectors may need update.")

        print("🚪 Close the browser window to end session.")
        await context.close()


if __name__ == "__main__":
    asyncio.run(main())

