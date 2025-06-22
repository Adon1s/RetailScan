"""
Temu Product Viability Screener
Initial screening to determine if products are worth deeper Amazon research

Features:
- Incremental saving: Each result is saved immediately after analysis
- Resume support: Skips already-analyzed products if script is restarted
- No duplicates: Checks existing results before analyzing
"""
import fire
import base64
import requests
import json
from pathlib import Path
from typing import Dict, List, Optional
import time
import warnings
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')


# Configuration
LM_STUDIO_URL = "http://192.168.86.143:1234"
PRODUCTS_JSON = Path("temu_products_for_analysis.json")
IMAGES_DIR = Path("temu_baby_toys_imgs")
ANALYSIS_OUTPUT = Path("viability_screening_results.json")

def create_analysis_prompt(product: Dict) -> str:
    """Create a prompt for initial viability screening"""
    prompt = f"""Analyze this Temu product for initial resale viability:

PRODUCT DETAILS:
- Title: {product['title']}
- Temu Cost: ${product['price']:.2f}
- Price Category: {product['price_category']}

KEYWORD INDICATORS:
- High Value Keywords: {', '.join(product.get('high_value_keywords', [])) or 'None'}
- Caution Keywords: {', '.join(product.get('caution_keywords', [])) or 'None'}
- Is Educational: {product.get('is_educational', False)}
- Is Bundle/Set: {product.get('is_bundle_or_set', False)}
- Has Gift Potential: {product.get('has_gift_potential', False)}

Based on the product image and details, provide:

1. PRODUCT QUALITY (1-10): Visual assessment of build quality and materials
2. MARKET DEMAND (1-10): How desirable is this type of product?
3. TARGET AUDIENCE: Primary buyers (be specific: age of kids, type of parents)
4. COMPETITION LEVEL: Is this a saturated product type? (Low/Medium/High)
5. SHIPPING FEASIBILITY: Size/weight concerns for online selling?
6. SAFETY/COMPLIANCE RISKS: Any potential safety or regulatory concerns?
7. BRAND POTENTIAL: Generic or could build brand around it?
8. VIABILITY SCORE (1-10): Worth researching further on Amazon?
9. PRIMARY CONCERNS: Main risks or red flags
10. VERDICT: RESEARCH FURTHER or SKIP (with brief reasoning)

Focus on whether this product is worth deeper market research, NOT on specific pricing."""

    return prompt

def analyze_product_local(product: Dict, image_path: Path, model_name: Optional[str] = None) -> Optional[str]:
    """Screen a product using local LM Studio SDK"""
    try:
        import lmstudio as lms

        model = lms.llm(model_name) if model_name else lms.llm()
        image_handle = lms.prepare_image(str(image_path))

        prompt = create_analysis_prompt(product)

        chat = lms.Chat()
        chat.add_user_message(prompt, images=[image_handle])

        prediction = model.respond(chat)
        return prediction
    except Exception as e:
        print(f"Local SDK error: {e}")
        return None

def analyze_product_remote(product: Dict, image_path: Path, model_name: Optional[str] = None) -> str:
    """Screen a product using remote LM Studio API"""
    # Read and encode image
    with open(image_path, 'rb') as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')

    prompt = create_analysis_prompt(product)

    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }],
        "max_tokens": 1000,
        "temperature": 0.7,
        "stream": False
    }

    if model_name:
        payload["model"] = model_name

    response = requests.post(
        f"{LM_STUDIO_URL}/v1/chat/completions",
        json=payload,
        timeout=60
    )

    result = response.json()
    return result['choices'][0]['message']['content']

def analyze_single_product(product: Dict, model_name: Optional[str] = None) -> Dict:
    """Screen a single product and return results"""
    temu_id = product['temu_id']
    image_path = IMAGES_DIR / f"{temu_id}.jpg"

    print(f"\n{'='*60}")
    print(f"Screening: {product['title'][:80]}...")
    print(f"Temu Cost: ${product['price']:.2f}")

    if not image_path.exists():
        print(f"⚠️  No image found for product {temu_id}")
        return {
            'temu_id': temu_id,
            'title': product['title'],
            'analysis': 'SKIPPED - No image available',
            'error': 'Image not found'
        }

    # Try local first, then remote
    analysis = analyze_product_local(product, image_path, model_name)
    if not analysis:
        print("Using remote LM Studio server...")
        analysis = analyze_product_remote(product, image_path, model_name)

    return {
        'temu_id': temu_id,
        'title': product['title'],
        'price': product['price'],
        'analysis': analysis,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

def load_products() -> Dict:
    """Load products from the organized JSON file"""
    with open(PRODUCTS_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_existing_results() -> Dict[str, Dict]:
    """Load existing analysis results to avoid duplicates"""
    if not ANALYSIS_OUTPUT.exists():
        return {}

    try:
        with open(ANALYSIS_OUTPUT, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Create a dict keyed by temu_id for easy lookup
            return {result['temu_id']: result for result in data.get('results', [])}
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")
        return {}

def append_result_to_file(result: Dict, existing_results: Dict[str, Dict]):
    """Append a single result to the output file"""
    # Add to existing results
    existing_results[result['temu_id']] = result

    # Prepare output data
    output_data = {
        'screening_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_screened': len(existing_results),
        'results': list(existing_results.values())
    }

    # Save to file (overwrites with updated data)
    with open(ANALYSIS_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved result for {result['temu_id']}")

def analyze_products(
    limit: int = 10,
    min_price: float = 0,
    max_price: float = 1000,
    category: Optional[str] = None,
    model_name: Optional[str] = None
):
    """
    Screen Temu products for initial viability before Amazon research

    :param limit: Maximum number of products to analyze
    :param min_price: Minimum price filter
    :param max_price: Maximum price filter
    :param category: Price category filter (low_margin, medium_margin, good_margin, high_ticket)
    :param model_name: Specific LM Studio model to use
    """
    print("\n=== Temu Product Viability Screener ===\n")

    # Load existing results to avoid duplicates
    existing_results = load_existing_results()
    print(f"Found {len(existing_results)} previously analyzed products")

    # Load products
    data = load_products()
    all_products = data['all_products']

    # Filter products
    filtered_products = []
    skipped_count = 0

    for product in all_products:
        # Skip if already analyzed
        if product['temu_id'] in existing_results:
            skipped_count += 1
            continue

        # Price filter
        if product['price'] < min_price or product['price'] > max_price:
            continue

        # Category filter
        if category and product.get('price_category') != category:
            continue

        # Must have image
        if not product.get('has_image'):
            continue

        filtered_products.append(product)

    if skipped_count > 0:
        print(f"Skipping {skipped_count} already-analyzed products")

    # Sort by high-value keywords (best prospects first)
    filtered_products.sort(
        key=lambda x: len(x.get('high_value_keywords', [])),
        reverse=True
    )

    # Limit products
    products_to_analyze = filtered_products[:limit]

    print(f"Found {len(filtered_products)} new products matching criteria")
    print(f"Screening top {len(products_to_analyze)} products...\n")

    if len(products_to_analyze) == 0:
        print("No new products to analyze!")
        return

    # Analyze each product
    analyzed_count = 0
    for i, product in enumerate(products_to_analyze, 1):
        print(f"\n[{i}/{len(products_to_analyze)}] Screening product...")

        try:
            result = analyze_single_product(product, model_name)

            # Print analysis
            print("\n📊 SCREENING RESULT:")
            print("-" * 60)
            print(result['analysis'])

            # Save immediately
            append_result_to_file(result, existing_results)
            analyzed_count += 1

        except Exception as e:
            print(f"❌ Error analyzing product {product['temu_id']}: {e}")
            # Continue with next product even if one fails

        # Small delay to avoid overwhelming the API
        if i < len(products_to_analyze):
            time.sleep(2)

    # Summary
    print("\n" + "="*60)
    print("SCREENING COMPLETE!")
    print(f"✓ Screened {analyzed_count} new products")
    print(f"✓ Total products in database: {len(existing_results)}")
    print(f"✓ Results saved to: {ANALYSIS_OUTPUT}")
    print("\nNext step: Research products marked 'RESEARCH FURTHER' on Amazon")

def analyze_specific_product(temu_id: str, model_name: Optional[str] = None):
    """
    Screen a specific product by its Temu ID for viability

    :param temu_id: The Temu product ID
    :param model_name: Specific LM Studio model to use
    """
    print(f"\n=== Screening Specific Product: {temu_id} ===\n")

    # Load existing results
    existing_results = load_existing_results()

    # Check if already analyzed
    if temu_id in existing_results:
        print(f"⚠️  Product {temu_id} has already been analyzed!")
        print("\nExisting analysis:")
        print("-" * 60)
        print(existing_results[temu_id]['analysis'])
        return

    # Load products
    data = load_products()

    # Find the product
    product = None
    for p in data['all_products']:
        if p['temu_id'] == temu_id:
            product = p
            break

    if not product:
        print(f"❌ Product {temu_id} not found!")
        return

    # Analyze
    try:
        result = analyze_single_product(product, model_name)

        # Print analysis
        print("\n📊 SCREENING RESULT:")
        print("-" * 60)
        print(result['analysis'])

        # Save immediately
        append_result_to_file(result, existing_results)

    except Exception as e:
        print(f"❌ Error analyzing product: {e}")

def show_analysis_stats():
    """Show statistics about analyzed products"""
    existing_results = load_existing_results()

    if not existing_results:
        print("No products have been analyzed yet.")
        return

    print(f"\n=== Analysis Statistics ===")
    print(f"Total products analyzed: {len(existing_results)}")

    # Count verdicts
    research_further = 0
    skip = 0

    for result in existing_results.values():
        analysis_text = result.get('analysis', '').upper()
        if 'RESEARCH FURTHER' in analysis_text:
            research_further += 1
        elif 'SKIP' in analysis_text:
            skip += 1

    print(f"Products to research further: {research_further}")
    print(f"Products to skip: {skip}")
    print(f"Other/Unclear: {len(existing_results) - research_further - skip}")

    print(f"\nResults file: {ANALYSIS_OUTPUT}")

def show_remaining_products(limit: int = 20):
    """Show products that haven't been analyzed yet"""
    existing_results = load_existing_results()
    data = load_products()
    all_products = data['all_products']

    remaining = []
    for product in all_products:
        if product['temu_id'] not in existing_results and product.get('has_image'):
            remaining.append(product)

    # Sort by high-value keywords
    remaining.sort(key=lambda x: len(x.get('high_value_keywords', [])), reverse=True)

    print(f"\n=== Remaining Products to Analyze ===")
    print(f"Total unanalyzed products: {len(remaining)}")
    print(f"\nTop {min(limit, len(remaining))} products by keyword density:\n")

    for i, product in enumerate(remaining[:limit], 1):
        print(f"{i}. {product['temu_id']} - ${product['price']:.2f}")
        print(f"   {product['title'][:80]}...")
        print(f"   Keywords: {', '.join(product.get('high_value_keywords', [])) or 'None'}")
        print()

def export_research_candidates(output_file: str = "research_candidates.json"):
    """Export products marked 'RESEARCH FURTHER' to a separate file"""
    existing_results = load_existing_results()

    if not existing_results:
        print("No products have been analyzed yet.")
        return

    # Find products to research
    research_candidates = []
    for result in existing_results.values():
        analysis_text = result.get('analysis', '').upper()
        if 'RESEARCH FURTHER' in analysis_text:
            research_candidates.append({
                'temu_id': result['temu_id'],
                'title': result['title'],
                'price': result['price'],
                'analysis_summary': result['analysis'][:200] + '...'
            })

    # Save to file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'export_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_candidates': len(research_candidates),
            'candidates': research_candidates
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Exported {len(research_candidates)} research candidates to: {output_path}")
    print("These products are ready for Amazon price research!")

def main():
    """Main entry point with Fire CLI"""
    fire.Fire({
        'analyze': analyze_products,
        'analyze_one': analyze_specific_product,
        'stats': show_analysis_stats,
        'remaining': show_remaining_products,
        'export': export_research_candidates
    })

if __name__ == "__main__":
    main()