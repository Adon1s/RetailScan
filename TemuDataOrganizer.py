"""
Organize Temu data for LLM purchasability analysis
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


# Configuration
CSV_FILE = Path("temu_baby_toys.csv")
IMAGES_DIR = Path("temu_baby_toys_imgs")
OUTPUT_FILE = Path("temu_products_for_analysis.json")


def analyze_title_for_keywords(title):
    """Extract valuable keywords for resale SEO"""
    # Keywords that indicate high resale value in baby products
    high_value_keywords = [
        'educational', 'learning', 'development', 'montessori', 'wooden',
        'organic', 'musical', 'interactive', 'sensory', 'stem', 'eco-friendly',
        'gift', 'premium', 'deluxe', 'set', 'bundle', 'complete'
    ]

    # Keywords that might indicate lower margins
    caution_keywords = [
        'batteries not included', 'simple', 'basic', 'small', 'mini'
    ]

    title_lower = title.lower()

    return {
        'high_value_keywords': [kw for kw in high_value_keywords if kw in title_lower],
        'caution_keywords': [kw for kw in caution_keywords if kw in title_lower],
        'title_length': len(title),
        'word_count': len(title.split())
    }


def categorize_price_range(price):
    """Categorize price for resale margin analysis"""
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


def load_and_analyze_products():
    """Load products and add analysis fields for LLM"""
    products_analysis = []

    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Basic product info  ── accepts either old (`temu_id`) or new (`product_id`) header
            temu_id = row.get('temu_id') or row.get('product_id')
            title = row['title']
            price = float(row['price']) if row.get('price') else 0.0

            # Check if image exists locally
            image_exists = (IMAGES_DIR / f"{temu_id}.jpg").exists()

            # Analyze title for resale potential
            title_analysis = analyze_title_for_keywords(title)

            # Build product analysis object
            product = {
                'temu_id': temu_id,
                'title': title,
                'price': price,
                'product_url': row['product_url'],
                'has_image': image_exists,
                'image_path': f"temu_baby_toys_imgs/{temu_id}.jpg" if image_exists else None,

                # Analysis fields for LLM
                'price_category': categorize_price_range(price),
                'potential_markup_2x': price * 2 if price > 0 else "unknown",
                'potential_markup_3x': price * 3 if price > 0 else "unknown",
                'high_value_keywords': title_analysis['high_value_keywords'],
                'caution_keywords': title_analysis['caution_keywords'],
                'title_word_count': title_analysis['word_count'],
                'has_gift_potential': 'gift' in title.lower() or 'christmas' in title.lower(),
                'is_educational': any(kw in title.lower() for kw in ['educational', 'learning', 'development']),
                'is_bundle_or_set': any(kw in title.lower() for kw in ['set', 'bundle', 'pack', 'pcs', 'pieces']),

                # Quick scoring hints for LLM
                'resale_score_hints': {
                    'has_multiple_value_keywords': len(title_analysis['high_value_keywords']) >= 2,
                    'good_price_range': price > 10 and price < 50,
                    'has_caution_flags': len(title_analysis['caution_keywords']) > 0,
                    'image_available': image_exists
                }
            }

            products_analysis.append(product)

    return products_analysis


def create_llm_analysis_file(products):
    """Create the final JSON file for LLM analysis"""
    # Summary statistics for context
    total_products = len(products)
    products_with_images = sum(1 for p in products if p['has_image'])
    products_with_prices = sum(1 for p in products if p['price'] > 0)

    # Price distribution
    price_distribution = {}
    for p in products:
        category = p['price_category']
        price_distribution[category] = price_distribution.get(category, 0) + 1

    # Products sorted by potential value
    products_by_value_keywords = sorted(
        products,
        key=lambda x: len(x['high_value_keywords']),
        reverse=True
    )

    # Final structure for LLM
    analysis_data = {
        'metadata': {
            'source': 'Temu Baby Toys',
            'analysis_date': datetime.now().isoformat(),
            'total_products': total_products,
            'products_with_images': products_with_images,
            'products_with_prices': products_with_prices,
            'price_distribution': price_distribution,
            'analysis_purpose': 'Purchasability assessment for resale'
        },

        'analysis_guidelines': {
            'good_resale_indicators': [
                'Price between $10-50 (good margin potential)',
                'Educational or developmental focus',
                'Gift-ready or seasonal items',
                'Bundle or set products',
                'Has product image available',
                'Multiple high-value keywords'
            ],
            'caution_indicators': [
                'No price listed',
                'Very low price (under $10)',
                'Missing product images',
                'Generic or simple items',
                'Items requiring batteries'
            ]
        },

        'top_prospects': {
            'by_keyword_density': [
                {
                    'temu_id': p['temu_id'],
                    'title': p['title'][:100] + '...' if len(p['title']) > 100 else p['title'],
                    'price': p['price'],
                    'high_value_keywords': p['high_value_keywords'],
                    'quick_assessment': f"Price: ${p['price']}, Keywords: {len(p['high_value_keywords'])}, Has image: {p['has_image']}"
                }
                for p in products_by_value_keywords[:10]
            ]
        },

        'all_products': products
    }

    # Save the analysis file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)

    return analysis_data


def main():
    """Main function"""
    print("Temu Data Analyzer for Resale")
    print("=" * 50)

    # Load and analyze products
    print("Loading and analyzing products...")
    products = load_and_analyze_products()
    print(f"✓ Analyzed {len(products)} products")

    # Create LLM analysis file
    print("\nCreating LLM analysis file...")
    analysis_data = create_llm_analysis_file(products)

    print(f"\n✓ Analysis file saved to: {OUTPUT_FILE}")

    # Quick summary
    print("\nQuick Summary:")
    print(f"- Total products: {analysis_data['metadata']['total_products']}")
    print(f"- Products with images: {analysis_data['metadata']['products_with_images']}")
    print(f"- Products with prices: {analysis_data['metadata']['products_with_prices']}")

    print("\nPrice Distribution:")
    for category, count in analysis_data['metadata']['price_distribution'].items():
        print(f"  - {category}: {count} products")

    print("\nTop 3 prospects by keyword analysis:")
    for i, product in enumerate(analysis_data['top_prospects']['by_keyword_density'][:3], 1):
        print(f"\n{i}. {product['title']}")
        print(f"   {product['quick_assessment']}")

    print("\n💡 Feed 'temu_products_for_analysis.json' to your LLM for detailed purchasability analysis")


if __name__ == "__main__":
    main