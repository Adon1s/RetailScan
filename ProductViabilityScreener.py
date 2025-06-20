"""
Temu Product Viability Screener
Initial screening to determine if products are worth deeper Amazon research
"""
import fire
import base64
import requests
import json
from pathlib import Path
from typing import Dict, List, Optional
import time

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

    print(f"\n{'=' * 60}")
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


def save_analysis_results(results: List[Dict]):
    """Save screening results to JSON file"""
    output_data = {
        'screening_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_screened': len(results),
        'results': results
    }

    with open(ANALYSIS_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Screening results saved to: {ANALYSIS_OUTPUT}")


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

    # Load products
    data = load_products()
    all_products = data['all_products']

    # Filter products
    filtered_products = []
    for product in all_products:
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

    # Sort by high-value keywords (best prospects first)
    filtered_products.sort(
        key=lambda x: len(x.get('high_value_keywords', [])),
        reverse=True
    )

    # Limit products
    products_to_analyze = filtered_products[:limit]

    print(f"Found {len(filtered_products)} products matching criteria")
    print(f"Screening top {len(products_to_analyze)} products...\n")

    # Analyze each product
    results = []
    for i, product in enumerate(products_to_analyze, 1):
        print(f"\n[{i}/{len(products_to_analyze)}] Screening product...")

        result = analyze_single_product(product, model_name)
        results.append(result)

        # Print analysis
        print("\n📊 SCREENING RESULT:")
        print("-" * 60)
        print(result['analysis'])

        # Small delay to avoid overwhelming the API
        if i < len(products_to_analyze):
            time.sleep(2)

    # Save results
    save_analysis_results(results)

    # Summary
    print("\n" + "=" * 60)
    print("SCREENING COMPLETE!")
    print(f"✓ Screened {len(results)} products")
    print(f"✓ Results saved to: {ANALYSIS_OUTPUT}")
    print("\nNext step: Research products marked 'RESEARCH FURTHER' on Amazon")


def analyze_specific_product(temu_id: str, model_name: Optional[str] = None):
    """
    Screen a specific product by its Temu ID for viability

    :param temu_id: The Temu product ID
    :param model_name: Specific LM Studio model to use
    """
    print(f"\n=== Screening Specific Product: {temu_id} ===\n")

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
    result = analyze_single_product(product, model_name)

    # Print analysis
    print("\n📊 SCREENING RESULT:")
    print("-" * 60)
    print(result['analysis'])

    # Save single result
    save_analysis_results([result])


def main():
    """Main entry point with Fire CLI"""
    fire.Fire({
        'analyze': analyze_products,
        'analyze_one': analyze_specific_product
    })


if __name__ == "__main__":
    main()