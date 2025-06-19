#!/usr/bin/env python3
"""
Temu scraper using undetected-chromedriver to avoid CDC detection
"""
import time
import csv
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

import requests
from tqdm import tqdm

# First, try to import undetected_chromedriver
try:
    import undetected_chromedriver as uc
except ImportError:
    print("ERROR: undetected-chromedriver is required!")
    print("\nPlease install it:")
    print("1. pip install setuptools")
    print("2. pip install undetected-chromedriver")
    print("\nThen run this script again.")
    exit(1)

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

SEARCH_URL = "https://www.temu.com/search_result.html?search_key=baby%20toys&search_method=user"
OUT_DIR = Path("temu_baby_toys_imgs")
CSV_FILE = Path("temu_baby_toys.csv")
PROFILE_DIR = Path("ChromeTemuProfile")


def create_undetected_driver():
    """Create an undetected Chrome driver."""
    print("Creating undetected Chrome driver...")

    # Make profile directory
    PROFILE_DIR.mkdir(exist_ok=True)

    # Configure undetected Chrome
    options = uc.ChromeOptions()
    options.add_argument(f'--user-data-dir={PROFILE_DIR.absolute()}')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-setuid-sandbox')
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-blink-features=AutomationControlled')

    # Create the driver
    driver = uc.Chrome(
        options=options,
        version_main=None,  # Auto-detect version
        use_subprocess=True  # Better stability
    )

    return driver


def check_for_cdc(driver):
    """Check if CDC properties are present."""
    cdc_check = driver.execute_script("""
        const cdcProps = [];
        for (let prop in window) {
            if (prop.includes('cdc_')) cdcProps.push(prop);
        }
        return cdcProps;
    """)

    if cdc_check:
        print(f"⚠️  WARNING: CDC properties found: {cdc_check}")
    else:
        print("✓ No CDC properties detected - undetected mode working!")

    return len(cdc_check) == 0


def extract_products(driver) -> List[Dict]:
    """Extract products from the page."""
    products = []
    seen_ids = set()

    print("🔍 Extracting products...")

    # Wait for products
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='goods.html']"))
        )
    except:
        print("⚠️  No products found")
        return products

    # Find all product links
    links = driver.find_elements(By.CSS_SELECTOR, "a[href*='goods.html']")
    print(f"Found {len(links)} product links")

    for link in links:
        try:
            href = link.get_attribute('href')
            if not href:
                continue

            # Extract ID
            match = re.search(r'goods_id=(\d+)', href)
            if not match:
                continue

            product_id = match.group(1)
            if product_id in seen_ids:
                continue

            seen_ids.add(product_id)

            # Find product container
            container = link
            for _ in range(4):
                container = container.find_element(By.XPATH, '..')

                try:
                    # Find image
                    img = container.find_element(By.TAG_NAME, 'img')
                    img_url = img.get_attribute('src')

                    if img_url and 'http' in img_url:
                        # Title
                        title = "Product"
                        for sel in ['[class*="title"]', '[class*="name"]', 'h2', 'h3']:
                            try:
                                elem = container.find_element(By.CSS_SELECTOR, sel)
                                text = elem.text.strip()
                                if text and len(text) > 5:
                                    title = text
                                    break
                            except:
                                pass

                        # Price
                        price = 0.0
                        for sel in ['[class*="price"]']:
                            try:
                                elem = container.find_element(By.CSS_SELECTOR, sel)
                                text = elem.text
                                match = re.search(r'[\d,]+\.?\d*', text)
                                if match:
                                    price = float(match.group().replace(',', ''))
                                    break
                            except:
                                pass

                        products.append({
                            'temu_id': product_id,
                            'title': title[:100],
                            'price': price,
                            'image_url': img_url,
                            'product_url': href
                        })
                        break
                except:
                    pass
        except:
            continue

    return products


def scroll_page(driver):
    """Scroll to load all products."""
    print("📜 Scrolling to load products...")

    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_count = 0
    max_scrolls = 20

    while scroll_count < max_scrolls:
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        # Check new height
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            # Try one more time
            driver.execute_script("window.scrollBy(0, -500);")
            time.sleep(0.5)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

            final_height = driver.execute_script("return document.body.scrollHeight")
            if final_height == new_height:
                break

        last_height = new_height
        scroll_count += 1

        if scroll_count % 5 == 0:
            print(f"  Scrolled {scroll_count} times...")


def download_image(row: Dict, dest_dir: Path):
    """Download product image."""
    try:
        url = row["image_url"]
        fname = dest_dir / f"{row['temu_id']}.jpg"

        if fname.exists():
            return

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.temu.com/'
        }

        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        fname.write_bytes(resp.content)
    except Exception as e:
        print(f"✗ Failed: {row['temu_id']} - {e}")


def main():
    """Main function."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Temu Scraper - Undetected ChromeDriver Version")
    print("=" * 50)
    print("✓ No CDC properties")
    print("✓ Appears as real Chrome")
    print("✓ Persistent profile")
    print("")

    try:
        # Create undetected driver
        driver = create_undetected_driver()

        # Verify no CDC properties
        if not check_for_cdc(driver):
            print("⚠️  CDC properties detected - Temu might still detect this!")

        # Navigate to Temu
        print(f"\n🌐 Loading: {SEARCH_URL}")
        driver.get(SEARCH_URL)
        time.sleep(5)

        # Check current URL
        current_url = driver.current_url
        print(f"📍 Current URL: {current_url}")

        # Check if login is needed
        if 'login' in current_url.lower():
            print("\n🔐 Please log in manually")
            print("After logging in, press Enter to continue...")
            input()

            # Navigate back to search
            driver.get(SEARCH_URL)
            time.sleep(5)

        # Check page status
        page_source = driver.page_source
        if "check your connection" in page_source.lower():
            print("\n⚠️  Temu shows 'check your connection' - possible detection")
        elif "sold out" in page_source.lower() and links_count > 10:
            print("\n⚠️  Multiple 'sold out' messages - possible detection")
        else:
            print("✓ Page appears to be loading normally")

        # Scroll to load products
        scroll_page(driver)

        # Extract products
        products = extract_products(driver)

        if not products:
            print("\n❌ No products found!")
            driver.save_screenshot("no_products.png")
            with open("page_source.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print("Saved screenshot and page source for debugging")
            return

        print(f"\n✓ Found {len(products)} products")

        # Save CSV
        with CSV_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["temu_id", "title", "price", "image_url", "product_url"]
            )
            writer.writeheader()
            writer.writerows(products)
        print(f"✓ Saved CSV: {CSV_FILE}")

        # Download images
        print(f"\n🖼️  Downloading images...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            tasks = [executor.submit(download_image, row, OUT_DIR) for row in products]
            for _ in tqdm(as_completed(tasks), total=len(tasks), unit="img"):
                pass

        print(f"\n✅ Complete!")
        print(f"📁 Images: {OUT_DIR.resolve()}")
        print(f"📊 Data: {CSV_FILE.resolve()}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        input("\nPress Enter to close browser...")
        driver.quit()


if __name__ == "__main__":
    main()