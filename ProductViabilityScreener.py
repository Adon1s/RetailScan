"""
Temu Product Viability Screener - Structured Output Version
Initial screening to determine if products are worth deeper Amazon research

Features:
- Structured JSON output from LLM for easy querying
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
warnings.filterwarnings('ignore',  message='urllib3 v2 only supports OpenSSL')


# Configuration
LM_STUDIO_URL = "http://192.168.86.143:1234"
PRODUCTS_JSON = Path("temu_products_for_analysis.json")
IMAGES_DIR = Path("temu_baby_toys_imgs")
ANALYSIS_OUTPUT = Path("viability_screening_results.json")

def create_analysis_prompt(product: Dict) -> str:
    """Create a prompt for initial viability screening with structured output"""
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

Based on the product image and details, provide your analysis in the following JSON format. Be sure to output ONLY valid JSON with no additional text:

{{
  "product_quality_score": <number 1-10>,
  "product_quality_notes": "<brief notes on build quality and materials>",
  
  "market_demand_score": <number 1-10>,
  "market_demand_notes": "<brief notes on desirability>",
  
  "target_audience": {{
    "primary_buyers": "<e.g., Parents of toddlers 1-3>",
    "child_age_range": "<e.g., 6-36 months>",
    "buyer_profile": "<e.g., Quality-conscious parents, Gift buyers>"
  }},
  
  "competition_level": "<Low/Medium/High>",
  "competition_notes": "<brief notes on market saturation>",
  
  "shipping_feasibility": {{
    "rating": "<Easy/Moderate/Difficult>",
    "concerns": "<e.g., Size, weight, fragility issues>"
  }},
  
  "safety_compliance_risks": {{
    "risk_level": "<Low/Medium/High>",
    "specific_concerns": "<e.g., Small parts, certifications needed>"
  }},
  
  "brand_potential": {{
    "rating": "<Generic/Moderate/High>",
    "notes": "<Can this build a brand?>"
  }},
  
  "viability_score": <number 1-10>,
  "primary_concerns": ["<concern 1>", "<concern 2>", "<concern 3>"],
  
  "verdict": "<RESEARCH FURTHER or SKIP>",
  "verdict_reasoning": "<1-2 sentence explanation>",
  
  "estimated_amazon_price_range": {{
    "low": <number>,
    "high": <number>
  }},
  
  "key_selling_points": ["<point 1>", "<point 2>", "<point 3>"],
  
  "recommended_keywords": ["<keyword 1>", "<keyword 2>", "<keyword 3>"]
}}

Focus on whether this product is worth deeper market research. Output ONLY the JSON, no other text."""

    return prompt

def parse_llm_response(response: str) -> Dict:
    """Parse the LLM response and extract JSON"""
    try:
        # Try to parse the entire response as JSON
        return json.loads(response)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from the response
        # Look for opening and closing braces
        start = response.find('{')
        end = response.rfind('}')

        if start != -1 and end != -1:
            json_str = response[start:end+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # If all parsing fails, return a fallback structure
        return {
            "error": "Failed to parse LLM response",
            "raw_response": response,
            "verdict": "SKIP",
            "verdict_reasoning": "Unable to parse analysis"
        }

def analyze_product_local(product: Dict, image_path: Path, model_name: Optional[str] = None) -> Optional[Dict]:
    """Screen a product using local LM Studio SDK"""
    try:
        import lmstudio as lms

        model = lms.llm(model_name) if model_name else lms.llm()
        image_handle = lms.prepare_image(str(image_path))

        prompt = create_analysis_prompt(product)

        chat = lms.Chat()
        chat.add_user_message(prompt, images=[image_handle])

        prediction = model.respond(chat)
        return parse_llm_response(prediction)
    except Exception as e:
        print(f"Local SDK error: {e}")
        return None

def analyze_product_remote(product: Dict, image_path: Path, model_name: Optional[str] = None) -> Dict:
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
    raw_content = result['choices'][0]['message']['content']
    return parse_llm_response(raw_content)

def analyze_single_product(product: Dict, model_name: Optional[str] = None) -> Dict:
    """Screen a single product and return structured results"""
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
            'price': product['price'],
            'analysis': {
                'error': 'Image not found',
                'verdict': 'SKIP',
                'verdict_reasoning': 'No image available for analysis'
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    # Try local first, then remote
    analysis = analyze_product_local(product, image_path, model_name)
    if not analysis:
        print("Using remote LM Studio server...")
        analysis = analyze_product_remote(product, image_path, model_name)

    # Add product metadata to the result
    return {
        'temu_id': temu_id,
        'title': product['title'],
        'price': product['price'],
        'price_category': product.get('price_category'),
        'high_value_keywords': product.get('high_value_keywords', []),
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

    # Prepare output data with summary statistics
    all_results = list(existing_results.values())

    # Calculate summary stats
    research_count = sum(1 for r in all_results if r.get('analysis', {}).get('verdict') == 'RESEARCH FURTHER')
    skip_count = sum(1 for r in all_results if r.get('analysis', {}).get('verdict') == 'SKIP')
    avg_viability = sum(r.get('analysis', {}).get('viability_score', 0) for r in all_results) / len(all_results) if all_results else 0

    output_data = {
        'screening_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_screened': len(existing_results),
        'summary': {
            'research_further_count': research_count,
            'skip_count': skip_count,
            'average_viability_score': round(avg_viability, 2)
        },
        'results': all_results
    }

    # Save to file (overwrites with updated data)
    with open(ANALYSIS_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved result for {result['temu_id']}")

def print_analysis_summary(result: Dict):
    """Print a formatted summary of the analysis"""
    analysis = result.get('analysis', {})

    print("\n📊 SCREENING RESULT:")
    print("-" * 60)

    if 'error' in analysis:
        print(f"❌ Error: {analysis.get('error')}")
        print(f"Raw response: {analysis.get('raw_response', 'N/A')[:200]}...")
    else:
        print(f"✓ Viability Score: {analysis.get('viability_score', 'N/A')}/10")
        print(f"✓ Quality Score: {analysis.get('product_quality_score', 'N/A')}/10")
        print(f"✓ Market Demand: {analysis.get('market_demand_score', 'N/A')}/10")
        print(f"✓ Competition: {analysis.get('competition_level', 'N/A')}")
        print(f"✓ Target: {analysis.get('target_audience', {}).get('primary_buyers', 'N/A')}")

        print(f"\n📍 Verdict: {analysis.get('verdict', 'N/A')}")
        print(f"   Reason: {analysis.get('verdict_reasoning', 'N/A')}")

        if analysis.get('estimated_amazon_price_range'):
            price_range = analysis['estimated_amazon_price_range']
            print(f"\n💰 Est. Amazon Price: ${price_range.get('low', 0):.2f} - ${price_range.get('high', 0):.2f}")

        if analysis.get('primary_concerns'):
            print(f"\n⚠️  Concerns: {', '.join(analysis['primary_concerns'])}")

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
    print("\n=== Temu Product Viability Screener (Structured Output) ===\n")

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

            # Print analysis summary
            print_analysis_summary(result)

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
        # Print the structured analysis nicely
        result = existing_results[temu_id]
        print_analysis_summary(result)
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

        # Print analysis summary
        print_analysis_summary(result)

        # Save immediately
        append_result_to_file(result, existing_results)

    except Exception as e:
        print(f"❌ Error analyzing product: {e}")

def show_analysis_stats():
    """Show statistics about analyzed products with structured data"""
    existing_results = load_existing_results()

    if not existing_results:
        print("No products have been analyzed yet.")
        return

    print(f"\n=== Analysis Statistics ===")
    print(f"Total products analyzed: {len(existing_results)}")

    # Aggregate statistics from structured data
    viability_scores = []
    quality_scores = []
    demand_scores = []
    verdicts = {'RESEARCH FURTHER': 0, 'SKIP': 0, 'OTHER': 0}
    competition_levels = {'Low': 0, 'Medium': 0, 'High': 0}

    for result in existing_results.values():
        analysis = result.get('analysis', {})

        # Collect scores
        if 'viability_score' in analysis:
            viability_scores.append(analysis['viability_score'])
        if 'product_quality_score' in analysis:
            quality_scores.append(analysis['product_quality_score'])
        if 'market_demand_score' in analysis:
            demand_scores.append(analysis['market_demand_score'])

        # Count verdicts
        verdict = analysis.get('verdict', 'OTHER')
        if verdict in verdicts:
            verdicts[verdict] += 1
        else:
            verdicts['OTHER'] += 1

        # Count competition levels
        comp_level = analysis.get('competition_level', 'Unknown')
        if comp_level in competition_levels:
            competition_levels[comp_level] += 1

    # Calculate averages
    avg_viability = sum(viability_scores) / len(viability_scores) if viability_scores else 0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    avg_demand = sum(demand_scores) / len(demand_scores) if demand_scores else 0

    print(f"\n📊 Score Averages:")
    print(f"  - Viability: {avg_viability:.1f}/10")
    print(f"  - Quality: {avg_quality:.1f}/10")
    print(f"  - Market Demand: {avg_demand:.1f}/10")

    print(f"\n📍 Verdict Distribution:")
    print(f"  - Research Further: {verdicts['RESEARCH FURTHER']}")
    print(f"  - Skip: {verdicts['SKIP']}")
    print(f"  - Other/Unclear: {verdicts['OTHER']}")

    print(f"\n🏁 Competition Levels:")
    for level, count in competition_levels.items():
        if count > 0:
            print(f"  - {level}: {count}")

    print(f"\n📁 Results file: {ANALYSIS_OUTPUT}")

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

def export_research_candidates(output_file: str = "research_candidates.json", min_viability: float = 7.0):
    """Export products marked 'RESEARCH FURTHER' to a separate file with all structured data"""
    existing_results = load_existing_results()

    if not existing_results:
        print("No products have been analyzed yet.")
        return

    # Find products to research
    research_candidates = []
    for result in existing_results.values():
        analysis = result.get('analysis', {})
        verdict = analysis.get('verdict', '')
        viability_score = analysis.get('viability_score', 0)

        if verdict == 'RESEARCH FURTHER' and viability_score >= min_viability:
            # Include all structured data for further processing
            research_candidates.append({
                'temu_id': result['temu_id'],
                'title': result['title'],
                'temu_price': result['price'],
                'price_category': result.get('price_category'),
                'viability_score': viability_score,
                'quality_score': analysis.get('product_quality_score'),
                'demand_score': analysis.get('market_demand_score'),
                'estimated_amazon_price': analysis.get('estimated_amazon_price_range'),
                'target_audience': analysis.get('target_audience'),
                'competition_level': analysis.get('competition_level'),
                'key_selling_points': analysis.get('key_selling_points', []),
                'recommended_keywords': analysis.get('recommended_keywords', []),
                'primary_concerns': analysis.get('primary_concerns', []),
                'verdict_reasoning': analysis.get('verdict_reasoning')
            })

    # Sort by viability score
    research_candidates.sort(key=lambda x: x['viability_score'], reverse=True)

    # Save to file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'export_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_candidates': len(research_candidates),
            'min_viability_filter': min_viability,
            'candidates': research_candidates
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Exported {len(research_candidates)} research candidates to: {output_path}")
    print(f"  (Filtered by viability score >= {min_viability})")
    print("\nThese products are ready for Amazon price research!")

def query_products(field: str, operator: str = ">=", value: float = 0, limit: int = 10):
    """
    Query analyzed products by any numeric field

    :param field: Field to query (e.g., 'viability_score', 'quality_score', 'demand_score')
    :param operator: Comparison operator ('>', '>=', '<', '<=', '==')
    :param value: Value to compare against
    :param limit: Maximum results to show
    """
    existing_results = load_existing_results()

    if not existing_results:
        print("No products have been analyzed yet.")
        return

    # Map field names to their paths in the data structure
    field_map = {
        'viability_score': lambda r: r.get('analysis', {}).get('viability_score', 0),
        'quality_score': lambda r: r.get('analysis', {}).get('product_quality_score', 0),
        'demand_score': lambda r: r.get('analysis', {}).get('market_demand_score', 0),
        'price': lambda r: r.get('price', 0),
    }

    if field not in field_map:
        print(f"Unknown field: {field}")
        print(f"Available fields: {', '.join(field_map.keys())}")
        return

    # Filter results
    filtered = []
    for result in existing_results.values():
        field_value = field_map[field](result)

        if operator == '>=' and field_value >= value:
            filtered.append(result)
        elif operator == '>' and field_value > value:
            filtered.append(result)
        elif operator == '<=' and field_value <= value:
            filtered.append(result)
        elif operator == '<' and field_value < value:
            filtered.append(result)
        elif operator == '==' and field_value == value:
            filtered.append(result)

    # Sort by the queried field
    filtered.sort(key=lambda r: field_map[field](r), reverse=(operator in ['>=', '>']))

    # Display results
    print(f"\n=== Products where {field} {operator} {value} ===")
    print(f"Found {len(filtered)} products\n")

    for i, result in enumerate(filtered[:limit], 1):
        analysis = result.get('analysis', {})
        print(f"{i}. {result['title'][:60]}...")
        print(f"   Temu ID: {result['temu_id']}")
        print(f"   Price: ${result['price']:.2f}")
        print(f"   {field}: {field_map[field](result)}")
        print(f"   Verdict: {analysis.get('verdict', 'N/A')}")
        print()

def main():
    """Main entry point with Fire CLI"""
    fire.Fire({
        'analyze': analyze_products,
        'analyze_one': analyze_specific_product,
        'stats': show_analysis_stats,
        'remaining': show_remaining_products,
        'export': export_research_candidates,
        'query': query_products
    })

if __name__ == "__main__":
    main()