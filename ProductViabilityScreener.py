# Temu Product Viability Screener - Simplified Schema Version
# Initial screening to determine if products are worth deeper Amazon research

# Features:
# - Simplified structured JSON output via LM Studio `response_format` schema
# - Incremental saving: Each result is saved immediately after analysis
# - Resume support: Skips already-analyzed products if script is restarted
# - No duplicates: Checks existing results before analyzing
# - Enhanced error handling and debugging

import fire
import base64
import requests
import json
from pathlib import Path
from typing import Dict, Optional, List
import time
import warnings

warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

# Configuration
LM_STUDIO_URL = "http://192.168.86.143:1234"
PRODUCTS_JSON = Path("temu_products_for_analysis.json")
IMAGES_DIR = Path("temu_baby_toys_imgs")
ANALYSIS_OUTPUT = Path("viability_screening_results_new.json")

# Debug mode - set to True to see detailed field analysis
DEBUG_MODE = True

# Simplified JSON Schema for better completion rates
ANALYSIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "temu_viability",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "product_quality_score": {
                    "type": "integer",
                    "description": "1-10 rating of product quality"
                },
                "market_demand_score": {
                    "type": "integer",
                    "description": "1-10 rating of market demand"
                },
                "viability_score": {
                    "type": "integer",
                    "description": "1-10 overall viability rating"
                },
                "competition_level": {
                    "type": "string",
                    "enum": ["Low", "Medium", "High"]
                },
                "safety_risk_level": {
                    "type": "string",
                    "enum": ["Low", "Medium", "High"]
                },
                "primary_concerns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Top 3-5 concerns about this product"
                },
                "verdict": {
                    "type": "string",
                    "enum": ["RESEARCH FURTHER", "SKIP"]
                },
                "verdict_reasoning": {
                    "type": "string",
                    "description": "1-2 sentence explanation"
                },
                "estimated_amazon_price_range": {
                    "type": "object",
                    "properties": {
                        "low": {"type": "number"},
                        "high": {"type": "number"}
                    },
                    "required": ["low", "high"]
                },
                "key_selling_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-5 main selling points"
                },
                "recommended_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "5-10 Amazon search keywords"
                }
            },
            "required": [
                "product_quality_score",
                "market_demand_score",
                "viability_score",
                "competition_level",
                "safety_risk_level",
                "verdict",
                "verdict_reasoning",
                "primary_concerns",
                "estimated_amazon_price_range",
                "key_selling_points",
                "recommended_keywords"
            ]
        }
    }
}


def create_analysis_prompt(product: Dict) -> str:
    """Generate the analyst prompt for simplified schema"""
    return f"""You are an expert e-commerce analyst evaluating products for Amazon resale potential.

PRODUCT TO ANALYZE:
- Title: {product['title']}
- Temu Cost: ${product['price']:.2f}
- Price Category: {product['price_category']}
- High-Value Keywords: {', '.join(product.get('high_value_keywords', []) or ['None'])}
- Caution Keywords: {', '.join(product.get('caution_keywords', []) or ['None'])}
- Is Educational: {product.get('is_educational', False)}
- Is Bundle/Set: {product.get('is_bundle_or_set', False)}
- Has Gift Potential: {product.get('has_gift_potential', False)}

REQUIRED ANALYSIS:
1. product_quality_score (1-10): Rate perceived quality from image
2. market_demand_score (1-10): Rate market appeal and demand
3. viability_score (1-10): Overall business opportunity score
4. competition_level: "Low", "Medium", or "High"
5. safety_risk_level: "Low", "Medium", or "High" 
6. primary_concerns: List 3-5 main risks or issues
7. verdict: If (quality + demand + viability) >= 18 then "RESEARCH FURTHER" else "SKIP"
8. verdict_reasoning: 1-2 sentences explaining your verdict
9. estimated_amazon_price_range: {{low: X, high: Y}} realistic prices
10. key_selling_points: 3-5 main selling features
11. recommended_keywords: 5-10 Amazon search terms

You MUST provide ALL fields. Focus on actionable insights for resale decisions."""


def parse_llm_response(response: str) -> Dict:
    """Parse the LLM response JSON string into a Python dict with validation"""
    try:
        parsed = json.loads(response)

        # Debug: Check for missing fields using simplified schema
        if DEBUG_MODE:
            all_expected_fields = [
                'product_quality_score', 'market_demand_score', 'viability_score',
                'competition_level', 'safety_risk_level', 'primary_concerns',
                'verdict', 'verdict_reasoning', 'estimated_amazon_price_range',
                'key_selling_points', 'recommended_keywords'
            ]

            missing_fields = [f for f in all_expected_fields if f not in parsed]
            if missing_fields:
                print(f"⚠️  DEBUG - Missing fields in response: {missing_fields}")

        return parsed

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return {
            "error": "Invalid JSON",
            "raw_response": response[:500],
            "verdict": "SKIP",
            "verdict_reasoning": "Parsing failed"
        }


def analyze_product_remote(product: Dict, image_path: Path, model_name: Optional[str] = None) -> Dict:
    """Screen a product via LM Studio REST API with structured output enforcement"""
    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")

    prompt = create_analysis_prompt(product)

    messages = [
        {
            "role": "system",
            "content": "You are a meticulous e-commerce analyst. You must provide ALL fields defined in the JSON schema. Follow the structured output format exactly."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }
    ]

    payload = {
        "model": model_name or "gemma-3-27b-it",
        "messages": messages,
        "response_format": ANALYSIS_SCHEMA,
        "temperature": 0.3,
        "max_tokens": 1500,  # Sufficient for simplified schema
        "stream": False
    }

    try:
        if DEBUG_MODE:
            print(f"📡 Sending request to LM Studio API...")

        resp = requests.post(f"{LM_STUDIO_URL}/v1/chat/completions", json=payload, timeout=120)

        if DEBUG_MODE:
            print(f"📡 Response status: {resp.status_code}")

    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return {
            "error": f"Request failed: {e}",
            "raw_response": "",
            "verdict": "SKIP",
            "verdict_reasoning": "HTTP request error"
        }

    # Non-200 HTTP status
    if resp.status_code != 200:
        print(f"❌ HTTP error {resp.status_code}")
        return {
            "error": f"HTTP {resp.status_code}",
            "raw_response": resp.text[:500],
            "verdict": "SKIP",
            "verdict_reasoning": "LLM server returned error status"
        }

    # Try to parse JSON response
    try:
        data = resp.json()
    except ValueError:
        print(f"❌ Invalid JSON from server")
        return {
            "error": "Non-JSON response from server",
            "raw_response": resp.text[:500],
            "verdict": "SKIP",
            "verdict_reasoning": "Invalid JSON from LLM server"
        }

    # Ensure expected keys
    if "choices" not in data or not data["choices"]:
        print(f"❌ Malformed response - missing choices")
        return {
            "error": "Missing 'choices' in response JSON",
            "raw_response": json.dumps(data)[:500],
            "verdict": "SKIP",
            "verdict_reasoning": "Malformed response"
        }

    content = data["choices"][0]["message"].get("content", "")

    if DEBUG_MODE:
        print(f"📡 Response length: {len(content)} chars")
        # Print token usage if available
        if "usage" in data:
            usage = data["usage"]
            print(f"📊 Tokens - Prompt: {usage.get('prompt_tokens', 'N/A')}, "
                  f"Completion: {usage.get('completion_tokens', 'N/A')}, "
                  f"Total: {usage.get('total_tokens', 'N/A')}")

    return parse_llm_response(content)


def analyze_single_product(product: Dict, model_name: Optional[str] = None) -> Dict:
    """Screen a single product and return structured results"""
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
            'price': product['price'],
            'analysis': {
                'error': 'Image not found',
                'verdict': 'SKIP',
                'verdict_reasoning': 'No image available for analysis'
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    # Use REST API directly (local SDK doesn't support structured output)
    analysis = analyze_product_remote(product, image_path, model_name)

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
            return {r['temu_id']: r for r in data.get('results', [])}
    except Exception as e:
        print(f"⚠️  Warning: Could not load existing results: {e}")
        return {}


def append_result_to_file(result: Dict, existing_results: Dict[str, Dict]):
    """Append a single result to the output file"""
    existing_results[result['temu_id']] = result
    all_results = list(existing_results.values())

    # Calculate summary stats
    research_count = sum(1 for r in all_results if r.get('analysis', {}).get('verdict') == 'RESEARCH FURTHER')
    skip_count = sum(1 for r in all_results if r.get('analysis', {}).get('verdict') == 'SKIP')

    # Safe average calculation
    viability_scores = [r.get('analysis', {}).get('viability_score', 0) for r in all_results if 'analysis' in r]
    avg_viability = sum(viability_scores) / len(viability_scores) if viability_scores else 0

    output_data = {
        'screening_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_screened': len(all_results),
        'summary': {
            'research_further_count': research_count,
            'skip_count': skip_count,
            'average_viability_score': round(avg_viability, 2)
        },
        'results': all_results
    }

    with open(ANALYSIS_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved result for {result['temu_id']}")


def print_analysis_summary(result: Dict):
    """Print a formatted summary of the analysis"""
    analysis = result.get('analysis', {})

    print("\n📊 SCREENING RESULT:")
    print("-" * 60)

    if 'error' in analysis:
        print(f"❌ Error: {analysis.get('error')}")
        if DEBUG_MODE and 'raw_response' in analysis:
            print(f"Raw response preview: {analysis.get('raw_response', 'N/A')[:200]}...")
    else:
        # Basic scores
        print(f"✓ Viability Score: {analysis.get('viability_score', 'N/A')}/10")
        print(f"✓ Quality Score: {analysis.get('product_quality_score', 'N/A')}/10")
        print(f"✓ Market Demand: {analysis.get('market_demand_score', 'N/A')}/10")

        # Calculate total score
        quality = analysis.get('product_quality_score', 0)
        demand = analysis.get('market_demand_score', 0)
        viability = analysis.get('viability_score', 0)
        total_score = quality + demand + viability
        print(f"✓ Total Score: {total_score}/30")

        # Key details
        print(f"\n📍 Verdict: {analysis.get('verdict', 'N/A')}")
        print(f"   Reason: {analysis.get('verdict_reasoning', 'N/A')}")

        # Competition and safety
        print(f"🏁 Competition: {analysis.get('competition_level', 'N/A')}")
        print(f"⚠️  Safety Risk: {analysis.get('safety_risk_level', 'N/A')}")

        # Pricing
        if analysis.get('estimated_amazon_price_range'):
            price_range = analysis['estimated_amazon_price_range']
            temu_price = result['price']
            potential_margin = price_range.get('low', 0) - temu_price
            print(f"\n💰 Est. Amazon Price: ${price_range.get('low', 0):.2f} - ${price_range.get('high', 0):.2f}")
            print(f"   Potential Margin: ${potential_margin:.2f} ({potential_margin / temu_price * 100:.0f}%)")

        # Concerns
        if analysis.get('primary_concerns'):
            print(f"\n🚨 Concerns: {', '.join(analysis['primary_concerns'])}")

        # Selling points (optional - in debug mode)
        if DEBUG_MODE and analysis.get('key_selling_points'):
            print(f"\n✨ Top Selling Points: {len(analysis['key_selling_points'])} identified")


def analyze_products(
        limit: Optional[int] = None,
        min_price: float = 0,
        max_price: float = 1000,
        category: Optional[str] = None,
        model_name: Optional[str] = None
):
    """Screen Temu products for initial viability before Amazon research

    Args:
        limit: Maximum number of products to analyze
        min_price: Minimum price filter
        max_price: Maximum price filter
        category: Price category filter (low_margin, medium_margin, good_margin, high_ticket)
        model_name: Specific LM Studio model to use
    """
    print("\n=== Temu Product Viability Screener (Simplified Schema) ===\n")

    # Load existing results to avoid duplicates
    existing_results = load_existing_results()
    print(f"Found {len(existing_results)} previously analyzed products")

    # Load products
    data = load_products()
    all_products = data['all_products']

    # Filter products
    filtered = [
        p for p in all_products
        if p['temu_id'] not in existing_results
           and p.get('has_image')
           and min_price <= p['price'] <= max_price
           and (not category or p.get('price_category') == category)
    ]

    # Sort by high-value keywords (best prospects first)
    filtered.sort(key=lambda x: len(x.get('high_value_keywords', [])), reverse=True)

    # Limit products if specified
    to_analyze = filtered if limit is None else filtered[:limit]

    print(f"Found {len(filtered)} products matching criteria")
    print(f"Analyzing {len(to_analyze)} products...\n")

    if not to_analyze:
        print("No new products to analyze!")
        return

    # Analyze each product
    analyzed_count = 0
    error_count = 0

    for i, product in enumerate(to_analyze, 1):
        print(f"\n[{i}/{len(to_analyze)}] Processing...")

        try:
            result = analyze_single_product(product, model_name)

            # Check if analysis was successful
            if 'error' in result.get('analysis', {}):
                error_count += 1
            else:
                analyzed_count += 1

            # Print analysis summary
            print_analysis_summary(result)

            # Save immediately
            append_result_to_file(result, existing_results)

        except Exception as e:
            print(f"❌ Error analyzing product {product['temu_id']}: {e}")
            error_count += 1

        # Small delay to avoid overwhelming the API
        if i < len(to_analyze):
            time.sleep(2)

    # Summary
    print("\n" + "=" * 60)
    print("SCREENING COMPLETE!")
    print(f"✅ Successfully analyzed: {analyzed_count} products")
    print(f"❌ Errors: {error_count} products")
    print(f"📊 Total in database: {len(existing_results)}")
    print(f"📁 Results saved to: {ANALYSIS_OUTPUT}")
    print("\n💡 Next step: Use 'export' command to get products marked 'RESEARCH FURTHER'")


def analyze_specific_product(temu_id: str, model_name: Optional[str] = None):
    """Screen a specific product by its Temu ID"""
    existing_results = load_existing_results()

    if temu_id in existing_results:
        print(f"⚠️  Product {temu_id} has already been analyzed!")
        result = existing_results[temu_id]
        print_analysis_summary(result)
        return

    # Find product
    data = load_products()
    product = next((p for p in data['all_products'] if p['temu_id'] == temu_id), None)

    if not product:
        print(f"❌ Product {temu_id} not found!")
        return

    # Analyze
    try:
        result = analyze_single_product(product, model_name)
        print_analysis_summary(result)
        append_result_to_file(result, existing_results)
    except Exception as e:
        print(f"❌ Error analyzing product: {e}")


def show_analysis_stats():
    """Show detailed statistics about analyzed products"""
    existing_results = load_existing_results()

    if not existing_results:
        print("No products have been analyzed yet.")
        return

    print(f"\n=== Analysis Statistics ===")
    print(f"Total products analyzed: {len(existing_results)}")

    # Collect all data
    all_scores = {'viability': [], 'quality': [], 'demand': [], 'total': []}
    verdicts = {'RESEARCH FURTHER': 0, 'SKIP': 0, 'ERROR': 0}
    competition_levels = {'Low': 0, 'Medium': 0, 'High': 0}
    safety_risks = {'Low': 0, 'Medium': 0, 'High': 0}
    field_completeness = {}

    # Expected fields for completeness check - using simplified schema
    expected_fields = [
        'product_quality_score', 'market_demand_score', 'viability_score',
        'competition_level', 'safety_risk_level', 'primary_concerns',
        'verdict', 'verdict_reasoning', 'estimated_amazon_price_range',
        'key_selling_points', 'recommended_keywords'
    ]

    for field in expected_fields:
        field_completeness[field] = 0

    for result in existing_results.values():
        analysis = result.get('analysis', {})

        if 'error' in analysis:
            verdicts['ERROR'] += 1
            continue

        # Collect scores
        quality = analysis.get('product_quality_score', 0)
        demand = analysis.get('market_demand_score', 0)
        viability = analysis.get('viability_score', 0)

        if quality: all_scores['quality'].append(quality)
        if demand: all_scores['demand'].append(demand)
        if viability: all_scores['viability'].append(viability)
        if quality and demand and viability:
            all_scores['total'].append(quality + demand + viability)

        # Count verdicts
        verdict = analysis.get('verdict', 'ERROR')
        verdicts[verdict] = verdicts.get(verdict, 0) + 1

        # Count competition levels
        comp = analysis.get('competition_level')
        if comp: competition_levels[comp] = competition_levels.get(comp, 0) + 1

        # Count safety risks - using simplified field name
        safety = analysis.get('safety_risk_level')
        if safety: safety_risks[safety] = safety_risks.get(safety, 0) + 1

        # Check field completeness
        for field in expected_fields:
            if field in analysis and analysis[field]:
                field_completeness[field] += 1

    # Calculate averages
    print(f"\n📊 Score Averages:")
    for score_type, scores in all_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  - {score_type.capitalize()}: {avg:.1f}")

    print(f"\n📍 Verdict Distribution:")
    for verdict, count in verdicts.items():
        if count > 0:
            print(f"  - {verdict}: {count} ({count / len(existing_results) * 100:.1f}%)")

    print(f"\n🏁 Competition Levels:")
    for level, count in competition_levels.items():
        if count > 0:
            print(f"  - {level}: {count}")

    print(f"\n⚠️  Safety Risk Levels:")
    for risk, count in safety_risks.items():
        if count > 0:
            print(f"  - {risk}: {count}")

    # Field completeness report
    print(f"\n📋 Field Completeness Report:")
    complete_products = len(existing_results) - verdicts['ERROR']
    for field, count in sorted(field_completeness.items(), key=lambda x: x[1], reverse=True):
        if complete_products > 0:
            percentage = count / complete_products * 100
            status = "✅" if percentage > 90 else "⚠️" if percentage > 50 else "❌"
            print(f"  {status} {field}: {count}/{complete_products} ({percentage:.1f}%)")


def show_remaining_products(limit: int = 20):
    """Show products that haven't been analyzed yet"""
    existing_results = load_existing_results()
    data = load_products()

    remaining = [
        p for p in data['all_products']
        if p['temu_id'] not in existing_results and p.get('has_image')
    ]

    # Sort by high-value keywords
    remaining.sort(key=lambda x: len(x.get('high_value_keywords', [])), reverse=True)

    print(f"\n=== Remaining Products to Analyze ===")
    print(f"Total unanalyzed products with images: {len(remaining)}")
    print(f"\nTop {min(limit, len(remaining))} products by keyword density:\n")

    for i, product in enumerate(remaining[:limit], 1):
        keywords = product.get('high_value_keywords', [])
        print(f"{i}. ID: {product['temu_id']} - ${product['price']:.2f}")
        print(f"   {product['title'][:80]}...")
        print(f"   Keywords ({len(keywords)}): {', '.join(keywords[:5])}")
        if len(keywords) > 5:
            print(f"   ... and {len(keywords) - 5} more")
        print()


def export_research_candidates(
        output_file: str = "research_candidates.json",
        min_viability: float = 7.0,
        include_all_data: bool = True
):
    """Export products marked 'RESEARCH FURTHER' to a separate file

    Args:
        output_file: Output filename
        min_viability: Minimum viability score filter
        include_all_data: Whether to include all analysis data or just basics
    """
    existing_results = load_existing_results()

    if not existing_results:
        print("No products have been analyzed yet.")
        return

    # Find products to research
    candidates = []

    for result in existing_results.values():
        analysis = result.get('analysis', {})

        if 'error' in analysis:
            continue

        verdict = analysis.get('verdict', '')
        viability_score = analysis.get('viability_score', 0)

        if verdict == 'RESEARCH FURTHER' and viability_score >= min_viability:
            if include_all_data:
                # Include comprehensive data
                candidate_data = {
                    'temu_id': result['temu_id'],
                    'title': result['title'],
                    'temu_price': result['price'],
                    'price_category': result.get('price_category'),
                    'scores': {
                        'viability': viability_score,
                        'quality': analysis.get('product_quality_score'),
                        'demand': analysis.get('market_demand_score'),
                        'total': (analysis.get('product_quality_score', 0) +
                                  analysis.get('market_demand_score', 0) +
                                  viability_score)
                    },
                    'estimated_amazon_price': analysis.get('estimated_amazon_price_range'),
                    'potential_margin': {
                        'low': analysis.get('estimated_amazon_price_range', {}).get('low', 0) - result['price'],
                        'high': analysis.get('estimated_amazon_price_range', {}).get('high', 0) - result['price']
                    },
                    'competition_level': analysis.get('competition_level'),
                    'safety_risk': analysis.get('safety_risk_level'),  # Using simplified field name
                    'key_selling_points': analysis.get('key_selling_points', []),
                    'recommended_keywords': analysis.get('recommended_keywords', []),
                    'primary_concerns': analysis.get('primary_concerns', []),
                    'verdict_reasoning': analysis.get('verdict_reasoning')
                }
            else:
                # Basic data only
                candidate_data = {
                    'temu_id': result['temu_id'],
                    'title': result['title'],
                    'viability_score': viability_score
                }

            candidates.append(candidate_data)

    # Sort by viability score (descending)
    candidates.sort(key=lambda x: x.get('viability_score', 0) if not include_all_data
    else x['scores']['viability'], reverse=True)

    # Save to file
    output_path = Path(output_file)
    output_data = {
        'export_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_candidates': len(candidates),
        'filters': {
            'min_viability_score': min_viability,
            'verdict': 'RESEARCH FURTHER'
        },
        'candidates': candidates
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Exported {len(candidates)} research candidates to: {output_path}")
    print(f"   Filtered by: viability score >= {min_viability}")

    # Show top candidates
    if candidates:
        print(f"\n🏆 Top 5 candidates by viability score:")
        for i, cand in enumerate(candidates[:5], 1):
            if include_all_data:
                margin = cand['potential_margin']
                print(f"{i}. {cand['title'][:60]}...")
                print(f"   Scores: V={cand['scores']['viability']}, "
                      f"Q={cand['scores']['quality']}, D={cand['scores']['demand']}")
                print(f"   Potential margin: ${margin['low']:.2f} - ${margin['high']:.2f}")
            else:
                print(f"{i}. Score {cand['viability_score']}: {cand['title'][:60]}...")


def query_products(field: str, operator: str = ">=", value: float = 0, limit: int = 10):
    """Query analyzed products by any numeric field"""
    existing_results = load_existing_results()

    if not existing_results:
        print("No products have been analyzed yet.")
        return

    # Map field names to their paths in the data structure
    field_map = {
        'viability_score': lambda r: r.get('analysis', {}).get('viability_score', 0),
        'quality_score': lambda r: r.get('analysis', {}).get('product_quality_score', 0),
        'demand_score': lambda r: r.get('analysis', {}).get('market_demand_score', 0),
        'total_score': lambda r: (r.get('analysis', {}).get('product_quality_score', 0) +
                                  r.get('analysis', {}).get('market_demand_score', 0) +
                                  r.get('analysis', {}).get('viability_score', 0)),
        'price': lambda r: r.get('price', 0),
        'margin_low': lambda r: (r.get('analysis', {}).get('estimated_amazon_price_range', {}).get('low', 0) -
                                 r.get('price', 0))
    }

    if field not in field_map:
        print(f"Unknown field: {field}")
        print(f"Available fields: {', '.join(field_map.keys())}")
        return

    # Filter results
    filtered = []
    for result in existing_results.values():
        # Skip error results
        if 'error' in result.get('analysis', {}):
            continue

        try:
            field_value = field_map[field](result)
            if eval(f"{field_value} {operator} {value}"):
                filtered.append(result)
        except:
            continue

    # Sort by the queried field
    filtered.sort(key=lambda r: field_map[field](r), reverse=(operator in ['>=', '>']))

    # Display results
    print(f"\n=== Products where {field} {operator} {value} ===")
    print(f"Found {len(filtered)} products\n")

    for i, result in enumerate(filtered[:limit], 1):
        analysis = result.get('analysis', {})
        field_value = field_map[field](result)

        print(f"{i}. {result['title'][:60]}...")
        print(f"   Temu ID: {result['temu_id']}")
        print(f"   Price: ${result['price']:.2f}")
        print(f"   {field}: {field_value:.2f}")
        print(f"   Verdict: {analysis.get('verdict', 'N/A')}")

        # Show scores for context
        if field not in ['quality_score', 'demand_score', 'viability_score', 'total_score']:
            quality = analysis.get('product_quality_score', 0)
            demand = analysis.get('market_demand_score', 0)
            viability = analysis.get('viability_score', 0)
            print(f"   Scores: Q={quality}, D={demand}, V={viability}, Total={quality + demand + viability}")
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