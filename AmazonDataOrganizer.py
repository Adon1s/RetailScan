"""
Organize Amazon data for LLM purchasability analysis
Creates a single JSON file optimized for resale decision making
"""
import csv
import json
from pathlib import Path
from datetime import datetime
import sys

# Force UTF-8 output so Windows console won’t choke on ✓, ✗, etc.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── CONFIG ──────────────────────────────────────────────────────────────
CSV_FILE     = Path("amazon_baby_toys.csv")
IMAGES_DIR   = Path("amazon_baby_toys_imgs")
OUTPUT_FILE  = Path("amazon_products_for_analysis.json")
ID_FIELD     = "amazon_id"            # only real schema change
SOURCE_NAME  = "Amazon Baby Toys"     # goes into metadata


# ── HELPER FUNCTIONS (unchanged) ─────────────────────────────────────────
def analyze_title_for_keywords(title):
    high_value_keywords = [
        "educational", "learning", "development", "montessori", "wooden",
        "organic", "musical", "interactive", "sensory", "stem", "eco-friendly",
        "gift", "premium", "deluxe", "set", "bundle", "complete"
    ]
    caution_keywords = [
        "batteries not included", "simple", "basic", "small", "mini"
    ]
    tl = title.lower()
    return {
        "high_value_keywords": [kw for kw in high_value_keywords if kw in tl],
        "caution_keywords":   [kw for kw in caution_keywords if kw in tl],
        "title_length": len(title),
        "word_count": len(title.split())
    }


def categorize_price_range(price):
    if price == 0:
        return "unknown"
    elif price < 10:
        return "low_margin"
    elif price < 25:
        return "medium_margin"
    elif price < 50:
        return "good_margin"
    else:
        return "high_ticket"


# ── CORE LOGIC ───────────────────────────────────────────────────────────
def load_and_analyze_products():
    products = []

    with CSV_FILE.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # accept either old (“amazon_id”) or new (“product_id”) column
            prod_id = row.get("amazon_id") or row.get("product_id")
            title = row["title"]
            price = float(row["price"]) if row.get("price") else 0.0

            img_exists = (IMAGES_DIR / f"{prod_id}.jpg").exists()
            title_info = analyze_title_for_keywords(title)

            products.append({
                ID_FIELD:         prod_id,
                "title":          title,
                "price":          price,
                "product_url":    row["product_url"],
                "has_image":      img_exists,
                "image_path":     f"{IMAGES_DIR.name}/{prod_id}.jpg" if img_exists else None,

                # derived fields
                "price_category":     categorize_price_range(price),
                "potential_markup_2x": price * 2 if price else "unknown",
                "potential_markup_3x": price * 3 if price else "unknown",
                **title_info,

                "has_gift_potential": "gift" in title.lower() or "christmas" in title.lower(),
                "is_educational":     any(k in title.lower() for k in ["educational", "learning", "development"]),
                "is_bundle_or_set":   any(k in title.lower() for k in ["set", "bundle", "pack", "pcs", "pieces"]),

                "resale_score_hints": {
                    "has_multiple_value_keywords": len(title_info["high_value_keywords"]) >= 2,
                    "good_price_range": 10 < price < 50,
                    "has_caution_flags": bool(title_info["caution_keywords"]),
                    "image_available":   img_exists,
                },
            })

    return products


def create_llm_analysis_file(products):
    total   = len(products)
    img_cnt = sum(p["has_image"]          for p in products)
    priced  = sum(p["price"] > 0          for p in products)

    price_distribution = {}
    for p in products:
        price_distribution[p["price_category"]] = price_distribution.get(p["price_category"], 0) + 1

    products_by_kw = sorted(products, key=lambda x: len(x["high_value_keywords"]), reverse=True)

    analysis = {
        "metadata": {
            "source": SOURCE_NAME,
            "analysis_date": datetime.now().isoformat(),
            "total_products": total,
            "products_with_images": img_cnt,
            "products_with_prices": priced,
            "price_distribution": price_distribution,
            "analysis_purpose": "Purchasability assessment for resale"
        },
        "analysis_guidelines": {        # unchanged text
            "good_resale_indicators": [
                "Price between $10-50 (good margin potential)",
                "Educational or developmental focus",
                "Gift-ready or seasonal items",
                "Bundle or set products",
                "Has product image available",
                "Multiple high-value keywords"
            ],
            "caution_indicators": [
                "No price listed",
                "Very low price (under $10)",
                "Missing product images",
                "Generic or simple items",
                "Items requiring batteries"
            ],
        },
        "top_prospects": {
            "by_keyword_density": [
                {
                    ID_FIELD: p[ID_FIELD],
                    "title": p["title"][:100] + "…" if len(p["title"]) > 100 else p["title"],
                    "price": p["price"],
                    "high_value_keywords": p["high_value_keywords"],
                    "quick_assessment": f"Price: ${p['price']}, Keywords: {len(p['high_value_keywords'])}, Image: {p['has_image']}"
                }
                for p in products_by_kw[:10]
            ]
        },
        "all_products": products
    }

    OUTPUT_FILE.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    return analysis


def main():
    print("Amazon Data Analyzer for Resale")
    print("="*50)

    print("Loading & analyzing products …")
    products = load_and_analyze_products()
    print(f" Analyzed {len(products)} products")

    print("\nCreating LLM analysis file …")
    analysis = create_llm_analysis_file(products)
    print(f"\n Analysis file saved → {OUTPUT_FILE}")

    # mini-summary
    meta = analysis["metadata"]
    print(f"\nQuick Summary:")
    print(f"- Total products:          {meta['total_products']}")
    print(f"- Products with images:    {meta['products_with_images']}")
    print(f"- Products with prices:    {meta['products_with_prices']}")
    print("\nPrice distribution:")
    for cat, n in meta["price_distribution"].items():
        print(f"  • {cat:13}: {n}")

    print("\nTop 3 prospects by keyword density:")
    for i, p in enumerate(analysis["top_prospects"]["by_keyword_density"][:3], 1):
        print(f"\n{i}. {p['title']}")
        print(f"   {p['quick_assessment']}")

    print(f"\n💡  Feed '{OUTPUT_FILE.name}' to your LLM for detailed purchasability analysis")


if __name__ == "__main__":
    main()