"""
Temu Product Analysis Visual Comparison Tool
Creates an interactive HTML dashboard for comparing analyzed products
"""
import json
import base64
from pathlib import Path
from typing import Dict, List
import webbrowser
import tempfile
import os


def load_analysis_results(filepath: Path = Path("viability_screening_results.json")):
    """Load the analysis results"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def encode_image(image_path: Path) -> str:
    """Encode image to base64 for embedding in HTML"""
    if not image_path.exists():
        return ""

    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def get_score_color(score: float) -> str:
    """Return color based on score (1-10 scale)"""
    if score >= 8:
        return "#22c55e"  # green
    elif score >= 6:
        return "#f59e0b"  # amber
    elif score >= 4:
        return "#ef4444"  # red
    else:
        return "#991b1b"  # dark red


def get_verdict_style(verdict: str) -> tuple:
    """Return color and emoji for verdict"""
    if verdict == "RESEARCH FURTHER":
        return "#22c55e", "✅"
    elif verdict == "SKIP":
        return "#ef4444", "❌"
    else:
        return "#6b7280", "❓"


def generate_product_card(product: Dict, images_dir: Path) -> str:
    """Generate HTML for a single product card"""
    temu_id = product['temu_id']
    analysis = product.get('analysis', {})

    # Get image
    image_path = images_dir / f"{temu_id}.jpg"
    image_data = encode_image(image_path)
    image_html = f'<img src="data:image/jpeg;base64,{image_data}" alt="Product">' if image_data else '<div class="no-image">No Image</div>'

    # Get scores
    viability = analysis.get('viability_score', 0)
    quality = analysis.get('product_quality_score', 0)
    demand = analysis.get('market_demand_score', 0)

    # Get verdict styling
    verdict = analysis.get('verdict', 'UNKNOWN')
    verdict_color, verdict_emoji = get_verdict_style(verdict)

    # Calculate profit
    price = product['price']
    est_low = analysis.get('estimated_amazon_price_range', {}).get('low', 0)
    est_high = analysis.get('estimated_amazon_price_range', {}).get('high', 0)
    profit_low = est_low - price
    profit_high = est_high - price
    roi_high = (profit_high / price * 100) if price > 0 else 0

    # Get other details
    competition = analysis.get('competition_level', 'Unknown')
    comp_color = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}.get(competition, "#6b7280")

    safety_risk = analysis.get('safety_compliance_risks', {}).get('risk_level', 'Unknown')
    safety_color = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}.get(safety_risk, "#6b7280")

    concerns = analysis.get('primary_concerns', [])
    keywords = product.get('high_value_keywords', [])

    card_html = f'''
    <div class="product-card" data-viability="{viability}" data-profit="{profit_high}" data-verdict="{verdict}">
        <div class="image-container">
            {image_html}
            <div class="price-badge">${price:.2f}</div>
            <div class="verdict-badge" style="background-color: {verdict_color}">
                {verdict_emoji} {verdict}
            </div>
        </div>

        <div class="product-info">
            <h3 class="product-title" title="{product['title']}">{product['title'][:60]}...</h3>

            <div class="scores-grid">
                <div class="score-item">
                    <span class="score-label">Viability</span>
                    <span class="score-value" style="background-color: {get_score_color(viability)}">{viability}/10</span>
                </div>
                <div class="score-item">
                    <span class="score-label">Quality</span>
                    <span class="score-value" style="background-color: {get_score_color(quality)}">{quality}/10</span>
                </div>
                <div class="score-item">
                    <span class="score-label">Demand</span>
                    <span class="score-value" style="background-color: {get_score_color(demand)}">{demand}/10</span>
                </div>
            </div>

            <div class="profit-section">
                <div class="profit-range">
                    💰 Est. Sell: ${est_low:.2f} - ${est_high:.2f}
                </div>
                <div class="profit-details">
                    Profit: ${profit_low:.2f} - ${profit_high:.2f} 
                    <span class="roi">(ROI: {roi_high:.0f}%)</span>
                </div>
            </div>

            <div class="details-grid">
                <div class="detail-item">
                    <span class="detail-label">Competition:</span>
                    <span class="detail-value" style="color: {comp_color}">{competition}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Safety Risk:</span>
                    <span class="detail-value" style="color: {safety_color}">{safety_risk}</span>
                </div>
            </div>

            <div class="keywords-section">
                <div class="keywords">
                    {' '.join([f'<span class="keyword">#{kw}</span>' for kw in keywords[:3]])}
                </div>
            </div>

            <div class="expandable-section">
                <button class="expand-btn" onclick="toggleDetails(this)">Show Details ▼</button>
                <div class="details-content" style="display: none;">
                    <div class="target-audience">
                        <strong>Target:</strong> {analysis.get('target_audience', {}).get('primary_buyers', 'N/A')}
                    </div>
                    <div class="concerns">
                        <strong>Concerns:</strong> {', '.join(concerns)}
                    </div>
                    <div class="reasoning">
                        <strong>Verdict Reasoning:</strong> {analysis.get('verdict_reasoning', 'N/A')}
                    </div>
                    <div class="selling-points">
                        <strong>Key Selling Points:</strong>
                        <ul>
                            {''.join([f'<li>{point}</li>' for point in analysis.get('key_selling_points', [])])}
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </div>
    '''

    return card_html


def generate_html_report(data: Dict, images_dir: Path) -> str:
    """Generate the complete HTML report"""
    products = data.get('results', [])

    # Sort products by viability score
    products.sort(key=lambda x: x.get('analysis', {}).get('viability_score', 0), reverse=True)

    # Generate cards
    product_cards = [generate_product_card(p, images_dir) for p in products]

    # Calculate summary stats
    total = len(products)
    research_count = sum(1 for p in products if p.get('analysis', {}).get('verdict') == 'RESEARCH FURTHER')
    skip_count = sum(1 for p in products if p.get('analysis', {}).get('verdict') == 'SKIP')

    html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Temu Product Analysis Dashboard</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background-color: #f3f4f6;
                color: #1f2937;
                line-height: 1.6;
            }}

            .header {{
                background-color: #1f2937;
                color: white;
                padding: 2rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}

            .header h1 {{
                font-size: 2rem;
                margin-bottom: 0.5rem;
            }}

            .summary-stats {{
                display: flex;
                gap: 2rem;
                margin-top: 1rem;
            }}

            .stat-item {{
                background-color: rgba(255,255,255,0.1);
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
            }}

            .controls {{
                background-color: white;
                padding: 1.5rem 2rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                display: flex;
                gap: 1rem;
                flex-wrap: wrap;
                align-items: center;
            }}

            .filter-group {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}

            select, button {{
                padding: 0.5rem 1rem;
                border: 1px solid #d1d5db;
                border-radius: 0.375rem;
                background-color: white;
                font-size: 0.875rem;
                cursor: pointer;
            }}

            button:hover {{
                background-color: #f3f4f6;
            }}

            .view-toggle {{
                margin-left: auto;
                display: flex;
                gap: 0.5rem;
            }}

            .products-container {{
                padding: 2rem;
            }}

            .products-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                gap: 1.5rem;
            }}

            .products-list {{
                display: none;
                flex-direction: column;
                gap: 1rem;
            }}

            .product-card {{
                background-color: white;
                border-radius: 0.75rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                overflow: hidden;
                transition: transform 0.2s, box-shadow 0.2s;
            }}

            .product-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}

            .list-view .product-card {{
                display: flex;
                height: 200px;
            }}

            .list-view .image-container {{
                width: 200px;
                height: 200px;
            }}

            .list-view .product-info {{
                flex: 1;
                padding: 1rem;
            }}

            .image-container {{
                position: relative;
                height: 200px;
                background-color: #f3f4f6;
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            .image-container img {{
                width: 100%;
                height: 100%;
                object-fit: cover;
            }}

            .no-image {{
                color: #9ca3af;
                font-size: 0.875rem;
            }}

            .price-badge {{
                position: absolute;
                top: 0.5rem;
                left: 0.5rem;
                background-color: #1f2937;
                color: white;
                padding: 0.25rem 0.5rem;
                border-radius: 0.375rem;
                font-weight: 500;
            }}

            .expand-btn {{
                background-color: #f3f4f6;
                border: none;
                padding: 0.375rem 0.75rem;
                border-radius: 0.375rem;
                font-size: 0.75rem;
                cursor: pointer;
                width: 100%;
                margin-top: 0.5rem;
            }}

            .expand-btn:hover {{
                background-color: #e5e7eb;
            }}

            .details-content {{
                margin-top: 0.75rem;
                padding-top: 0.75rem;
                border-top: 1px solid #e5e7eb;
                font-size: 0.8125rem;
            }}

            .details-content strong {{
                color: #4b5563;
                display: block;
                margin-bottom: 0.25rem;
            }}

            .details-content div {{
                margin-bottom: 0.5rem;
            }}

            .details-content ul {{
                margin-left: 1.5rem;
                margin-top: 0.25rem;
            }}

            @media (max-width: 768px) {{
                .products-grid {{
                    grid-template-columns: 1fr;
                }}

                .controls {{
                    flex-direction: column;
                    align-items: stretch;
                }}

                .view-toggle {{
                    margin-left: 0;
                    margin-top: 1rem;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Temu Product Analysis Dashboard</h1>
            <div class="summary-stats">
                <div class="stat-item">
                    <strong>Total Products:</strong> {total}
                </div>
                <div class="stat-item">
                    <strong>Research Further:</strong> {research_count}
                </div>
                <div class="stat-item">
                    <strong>Skip:</strong> {skip_count}
                </div>
            </div>
        </div>

        <div class="controls">
            <div class="filter-group">
                <label for="sort-select">Sort by:</label>
                <select id="sort-select" onchange="sortProducts()">
                    <option value="viability">Viability Score</option>
                    <option value="profit">Profit Potential</option>
                    <option value="demand">Market Demand</option>
                    <option value="quality">Quality Score</option>
                </select>
            </div>

            <div class="filter-group">
                <label for="filter-verdict">Filter:</label>
                <select id="filter-verdict" onchange="filterProducts()">
                    <option value="all">All Products</option>
                    <option value="RESEARCH FURTHER">Research Further</option>
                    <option value="SKIP">Skip</option>
                </select>
            </div>

            <div class="filter-group">
                <label for="min-viability">Min Viability:</label>
                <select id="min-viability" onchange="filterProducts()">
                    <option value="0">All</option>
                    <option value="6">6+</option>
                    <option value="7">7+</option>
                    <option value="8">8+</option>
                </select>
            </div>

            <div class="view-toggle">
                <button onclick="setView('grid')" id="grid-btn">Grid View</button>
                <button onclick="setView('list')" id="list-btn">List View</button>
            </div>
        </div>

        <div class="products-container">
            <div id="products-grid" class="products-grid">
                {''.join(product_cards)}
            </div>
        </div>

        <script>
            function toggleDetails(button) {{
                const details = button.nextElementSibling;
                if (details.style.display === 'none') {{
                    details.style.display = 'block';
                    button.textContent = 'Hide Details ▲';
                }} else {{
                    details.style.display = 'none';
                    button.textContent = 'Show Details ▼';
                }}
            }}

            function sortProducts() {{
                const container = document.getElementById('products-grid');
                const cards = Array.from(container.getElementsByClassName('product-card'));
                const sortBy = document.getElementById('sort-select').value;

                cards.sort((a, b) => {{
                    let aVal, bVal;

                    switch(sortBy) {{
                        case 'viability':
                            aVal = parseFloat(a.dataset.viability);
                            bVal = parseFloat(b.dataset.viability);
                            break;
                        case 'profit':
                            aVal = parseFloat(a.dataset.profit);
                            bVal = parseFloat(b.dataset.profit);
                            break;
                        case 'demand':
                            aVal = parseFloat(a.querySelector('.score-value:nth-child(3)').textContent);
                            bVal = parseFloat(b.querySelector('.score-value:nth-child(3)').textContent);
                            break;
                        case 'quality':
                            aVal = parseFloat(a.querySelector('.score-value:nth-child(2)').textContent);
                            bVal = parseFloat(b.querySelector('.score-value:nth-child(2)').textContent);
                            break;
                    }}

                    return bVal - aVal;
                }});

                cards.forEach(card => container.appendChild(card));
            }}

            function filterProducts() {{
                const verdictFilter = document.getElementById('filter-verdict').value;
                const minViability = parseFloat(document.getElementById('min-viability').value);
                const cards = document.getElementsByClassName('product-card');

                Array.from(cards).forEach(card => {{
                    const verdict = card.dataset.verdict;
                    const viability = parseFloat(card.dataset.viability);

                    let show = true;

                    if (verdictFilter !== 'all' && verdict !== verdictFilter) {{
                        show = false;
                    }}

                    if (viability < minViability) {{
                        show = false;
                    }}

                    card.style.display = show ? '' : 'none';
                }});
            }}

            function setView(view) {{
                const container = document.getElementById('products-grid');

                if (view === 'list') {{
                    container.classList.remove('products-grid');
                    container.classList.add('products-list');
                    document.body.classList.add('list-view');
                    document.getElementById('list-btn').style.backgroundColor = '#1f2937';
                    document.getElementById('list-btn').style.color = 'white';
                    document.getElementById('grid-btn').style.backgroundColor = 'white';
                    document.getElementById('grid-btn').style.color = '#1f2937';
                }} else {{
                    container.classList.remove('products-list');
                    container.classList.add('products-grid');
                    document.body.classList.remove('list-view');
                    document.getElementById('grid-btn').style.backgroundColor = '#1f2937';
                    document.getElementById('grid-btn').style.color = 'white';
                    document.getElementById('list-btn').style.backgroundColor = 'white';
                    document.getElementById('list-btn').style.color = '#1f2937';
                }}
            }}

            // Initialize
            document.getElementById('grid-btn').style.backgroundColor = '#1f2937';
            document.getElementById('grid-btn').style.color = 'white';
        </script>
    </body>
    </html>
    '''

    return html


def create_comparison_view(
        results_file: str = "viability_screening_results.json",
        images_dir: str = "temu_baby_toys_imgs",
        output_file: str = "product_comparison.html",
        auto_open: bool = True
):
    """
    Create an interactive HTML comparison view of analyzed products

    :param results_file: Path to the JSON results file
    :param images_dir: Directory containing product images
    :param output_file: Output HTML filename
    :param auto_open: Automatically open in browser
    """
    # Load data
    results_path = Path(results_file)
    images_path = Path(images_dir)

    if not results_path.exists():
        print(f"❌ Results file not found: {results_file}")
        return

    print(f"Loading results from: {results_file}")
    data = load_analysis_results(results_path)

    # Generate HTML
    print("Generating comparison view...")
    html_content = generate_html_report(data, images_path)

    # Save to file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ Comparison view saved to: {output_path}")

    # Open in browser if requested
    if auto_open:
        webbrowser.open(f'file://{output_path.absolute()}')
        print("📊 Opening in browser...")


def create_excel_comparison(
        results_file: str = "viability_screening_results.json",
        output_file: str = "product_comparison.xlsx"
):
    """
    Create an Excel file for detailed comparison

    :param results_file: Path to the JSON results file
    :param output_file: Output Excel filename
    """
    import pandas as pd

    # Load data
    results_path = Path(results_file)
    data = load_analysis_results(results_path)

    # Flatten the data for Excel
    rows = []
    for product in data.get('results', []):
        analysis = product.get('analysis', {})

        row = {
            'Temu ID': product['temu_id'],
            'Title': product['title'].replace('\n', ' ')[:100],
            'Temu Price': product['price'],
            'Category': product.get('price_category', ''),
            'Viability Score': analysis.get('viability_score', 0),
            'Quality Score': analysis.get('product_quality_score', 0),
            'Demand Score': analysis.get('market_demand_score', 0),
            'Verdict': analysis.get('verdict', ''),
            'Competition': analysis.get('competition_level', ''),
            'Safety Risk': analysis.get('safety_compliance_risks', {}).get('risk_level', ''),
            'Est. Min Price': analysis.get('estimated_amazon_price_range', {}).get('low', 0),
            'Est. Max Price': analysis.get('estimated_amazon_price_range', {}).get('high', 0),
            'Min Profit': analysis.get('estimated_amazon_price_range', {}).get('low', 0) - product['price'],
            'Max Profit': analysis.get('estimated_amazon_price_range', {}).get('high', 0) - product['price'],
            'ROI %': ((analysis.get('estimated_amazon_price_range', {}).get('high', 0) - product['price']) / product[
                'price'] * 100) if product['price'] > 0 else 0,
            'Target Audience': analysis.get('target_audience', {}).get('primary_buyers', ''),
            'Keywords': ', '.join(product.get('high_value_keywords', [])),
            'Primary Concerns': ', '.join(analysis.get('primary_concerns', [])),
            'Verdict Reasoning': analysis.get('verdict_reasoning', '')
        }
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Sort by viability score
    df = df.sort_values('Viability Score', ascending=False)

    # Save to Excel with formatting
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Product Analysis', index=False)

        # Get the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Product Analysis']

        # Adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter

            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass

            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    print(f"✅ Excel comparison saved to: {output_file}")


def main():
    """Main entry point"""
    import fire

    fire.Fire({
        'html': create_comparison_view,
        'excel': create_excel_comparison
    })


if __name__ == "__main__":
    main()
: 600;
font - size: 0.875
rem;
}}

.verdict - badge
{{
position: absolute;
top: 0.5
rem;
right: 0.5
rem;
color: white;
padding: 0.25
rem
0.5
rem;
border - radius: 0.375
rem;
font - weight: 600;
font - size: 0.75
rem;
}}

.product - info
{{
padding: 1
rem;
}}

.product - title
{{
font - size: 0.875
rem;
font - weight: 600;
margin - bottom: 0.75
rem;
overflow: hidden;
text - overflow: ellipsis;
white - space: nowrap;
}}

.scores - grid
{{
display: grid;
grid - template - columns: repeat(3, 1
fr);
gap: 0.5
rem;
margin - bottom: 0.75
rem;
}}

.score - item
{{
text - align: center;
}}

.score - label
{{
display: block;
font - size: 0.75
rem;
color:  # 6b7280;
margin - bottom: 0.25
rem;
}}

.score - value
{{
display: inline - block;
padding: 0.125
rem
0.5
rem;
border - radius: 0.375
rem;
color: white;
font - weight: 600;
font - size: 0.75
rem;
}}

.profit - section
{{
background - color:  # f9fafb;
padding: 0.5
rem;
border - radius: 0.375
rem;
margin - bottom: 0.75
rem;
font - size: 0.8125
rem;
}}

.profit - range
{{
font - weight: 600;
color:  # 059669;
margin - bottom: 0.25
rem;
}}

.profit - details
{{
color:  # 6b7280;
}}

.roi
{{
color:  # 1f2937;
font - weight: 600;
}}

.details - grid
{{
display: grid;
grid - template - columns: repeat(2, 1
fr);
gap: 0.5
rem;
margin - bottom: 0.75
rem;
font - size: 0.75
rem;
}}

.detail - label
{{
color:  # 6b7280;
}}

.detail - value
{{
font - weight: 600;
}}

.keywords - section
{{
margin - bottom: 0.75
rem;
}}

.keywords
{{
display: flex;
gap: 0.25
rem;
flex - wrap: wrap;
}}

.keyword
{{
background - color:  # dbeafe;
color:  # 1e40af;
padding: 0.125
rem
0.375
rem;
border - radius: 0.25
rem;
font - size: 0.6875
rem;
font - weight