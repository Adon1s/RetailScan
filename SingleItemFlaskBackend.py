"""
Flask Backend for Single Item Product Matcher
Provides API endpoints for the web dashboard
"""
from CORScanner.cors_scan import results
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import os
import tempfile
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
from werkzeug.utils import secure_filename
import re
import atexit
import shutil

# Import the single item matcher (make sure it's in the same directory)
from SingleItemMatcher import SingleItemMatcher, sanitize_filename

"""
Flask Backend for Single Item Product Matcher
Provides API endpoints for the web dashboard
"""
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import os
import tempfile
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
from werkzeug.utils import secure_filename
import re
import atexit
import shutil
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import the single item matcher (make sure it's in the same directory)
from SingleItemMatcher import SingleItemMatcher, sanitize_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Global matcher instance (reused for caching efficiency)
matcher = None

# Track temporary files for cleanup
temp_files_to_cleanup = set()


def cleanup_temp_files():
    """Clean up temporary files on exit"""
    for temp_file in temp_files_to_cleanup:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info(f"Cleaned up temp file: {temp_file}")
        except Exception as e:
            logger.error(f"Error cleaning up {temp_file}: {e}")
    temp_files_to_cleanup.clear()


# Register cleanup function
atexit.register(cleanup_temp_files)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def initialize_matcher(use_images=True):
    """Initialize or get the global matcher instance"""
    global matcher
    if matcher is None:
        matcher = SingleItemMatcher(use_images=use_images)
    return matcher


# Serve the dashboard HTML
@app.route('/')
def serve_dashboard():
    """Serve the main dashboard"""
    # Try to serve the existing HTML file
    if os.path.exists('ProductSearchDashboard.html'):
        return send_from_directory('.', 'ProductSearchDashboard.html')
    else:
        return "Dashboard HTML file not found. Please ensure ProductSearchDashboard.html is in the same directory.", 404


# Serve static files (for images, CSS, JS if needed)
@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    if os.path.exists(filename):
        return send_from_directory('.', filename)
    return "File not found", 404


# Create a static uploads directory
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Add route to serve uploaded files
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def run_scraper_with_input(scraper_script: str, search_terms: str, num_pages: int, platform: str):
    """Run a scraper with predefined inputs"""
    wrapper_file = Path(f"temp_{platform}_scraper_wrapper.py")

    wrapper_code = textwrap.dedent(f'''
        import subprocess
        import sys

        # Prepare inputs for the scraper
        inputs = "{search_terms}\\n{num_pages}\\n"

        # Run the actual scraper with inputs
        result = subprocess.run(
            [sys.executable, "{scraper_script}"],
            input=inputs,
            text=True,
            capture_output=True,
            encoding='utf-8'
        )

        # Forward the output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Exit with same code
        sys.exit(result.returncode)
    ''')

    try:
        with open(wrapper_file, 'w', encoding='utf-8') as f:
            f.write(wrapper_code)

        result = subprocess.run(
            [sys.executable, str(wrapper_file)],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        return result.returncode == 0, result.stdout, result.stderr

    finally:
        if wrapper_file.exists():
            wrapper_file.unlink()


def run_data_organizer(organizer_script: str, category: str, platform: str):
    """Run data organizer for a specific category"""
    wrapper_file = Path(f"temp_{platform}_organizer_wrapper.py")

    wrapper_code = textwrap.dedent(f'''
        import sys
        from pathlib import Path

        sys.path.insert(0, ".")

        if "{platform}" == "temu":
            import TemuDataOrganizer
            TemuDataOrganizer.CSV_FILE = Path("temu_{category}.csv")
            TemuDataOrganizer.IMAGES_DIR = Path("temu_{category}_imgs")
            TemuDataOrganizer.OUTPUT_FILE = Path("temu_{category}_analysis.json")
            TemuDataOrganizer.SOURCE_NAME = "{category}".replace('_', ' ').title()
            TemuDataOrganizer.main()
        else:
            import AmazonDataOrganizer
            AmazonDataOrganizer.CSV_FILE = Path("amazon_{category}.csv")
            AmazonDataOrganizer.IMAGES_DIR = Path("amazon_{category}_imgs")
            AmazonDataOrganizer.OUTPUT_FILE = Path("amazon_{category}_analysis.json")
            AmazonDataOrganizer.SOURCE_NAME = "{category}".replace('_', ' ').title()
            AmazonDataOrganizer.main()
    ''')

    try:
        with open(wrapper_file, 'w', encoding='utf-8') as f:
            f.write(wrapper_code)

        result = subprocess.run(
            [sys.executable, str(wrapper_file)],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        return result.returncode == 0

    finally:
        if wrapper_file.exists():
            wrapper_file.unlink()


@app.route('/api/single-item/scrape', methods=['POST'])
def scrape_fresh_data():
    """Run scrapers for fresh data"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        search_term = data.get('searchTerm', '').strip()
        category = data.get('category', '').strip()
        num_pages = data.get('numPages', 1)

        if not search_term:
            return jsonify({'error': 'Search term is required'}), 400

        if not category:
            return jsonify({'error': 'Category is required'}), 400

        # Sanitize category for file paths
        category_slug = sanitize_filename(category)

        logger.info(f"Starting fresh scrape - Search: {search_term}, Category: {category_slug}, Pages: {num_pages}")

        # Track results
        results = {
            'temu_success': False,
            'amazon_success': False,
            'temu_count': 0,
            'amazon_count': 0
        }

        # Run scrapers in parallel
        def run_scraper_thread(platform: str, script: str):
            try:
                success, stdout, stderr = run_scraper_with_input(
                    script,
                    search_term,
                    num_pages,
                    platform
                )

                if success:
                    # Count products from CSV
                    csv_file = Path(f"{platform}_{category_slug}.csv")
                    if csv_file.exists():
                        import csv
                        with open(csv_file, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            count = sum(1 for row in reader)
                            results[f'{platform}_count'] = count

                    results[f'{platform}_success'] = True
                    logger.info(f"{platform.capitalize()} scraping completed successfully")
                else:
                    logger.error(f"{platform.capitalize()} scraping failed: {stderr}")

            except Exception as e:
                logger.error(f"Error running {platform} scraper: {str(e)}")

        # Run scrapers in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(run_scraper_thread, 'temu', 'TemuScraper.py'): 'temu',
                executor.submit(run_scraper_thread, 'amazon', 'AmazonScraper.py'): 'amazon'
            }

            for future in as_completed(futures):
                platform = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Exception in {platform} scraper thread: {e}")

        # Check if at least one scraper succeeded
        if not results['temu_success'] and not results['amazon_success']:
            return jsonify({'error': 'Both scrapers failed'}), 500

        # Run data organizers
        logger.info("Running data organizers...")

        if results['temu_success']:
            if not run_data_organizer('TemuDataOrganizer.py', category_slug, 'temu'):
                logger.warning("Temu data organizer failed")

        if results['amazon_success']:
            if not run_data_organizer('AmazonDataOrganizer.py', category_slug, 'amazon'):
                logger.warning("Amazon data organizer failed")

        logger.info(f"Scraping completed - Temu: {results['temu_count']}, Amazon: {results['amazon_count']}")

        return jsonify({
            'success': True,
            'temu_count': results['temu_count'],
            'amazon_count': results['amazon_count'],
            'message': f"Scraped {results['temu_count'] + results['amazon_count']} total products"
        })

    except Exception as e:
        logger.error(f"Error in scrape endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Main matching endpoint
@app.route('/api/single-item/match', methods=['POST'])
def match_single_item():
    """Process single item matching request"""
    try:
        # Get form data
        title = request.form.get('title', '').strip()
        category = request.form.get('category', '').strip()
        use_images = request.form.get('useImages', 'true').lower() == 'true'
        top_n = int(request.form.get('topN', 20))

        if not title:
            return jsonify({'error': 'Product title is required'}), 400

        if not category:
            return jsonify({'error': 'Category is required'}), 400

        # Sanitize category for file paths
        category_slug = sanitize_filename(category)

        # Handle image input (file upload or URL)
        image_path = None
        temp_file_path = None
        web_accessible_path = None  # Add this

        # Check for uploaded file
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename and allowed_file(file.filename):
                # Save uploaded file to static directory
                filename = secure_filename(file.filename)
                timestamp_filename = f"upload_{datetime.now().timestamp()}_{filename}"
                temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], timestamp_filename)
                file.save(temp_file_path)
                image_path = temp_file_path
                # Store web-accessible path
                web_accessible_path = f"/static/uploads/{timestamp_filename}"
                logger.info(f"Saved uploaded image to: {temp_file_path}")

        # Check for image URL
        elif 'imageUrl' in request.form:
            image_url = request.form.get('imageUrl', '').strip()
            if image_url:
                image_path = image_url
                logger.info(f"Using image URL: {image_url}")

        if not image_path:
            return jsonify({'error': 'Product image is required (file upload or URL)'}), 400

        # Initialize matcher
        matcher_instance = initialize_matcher(use_images=use_images)

        # Log the matching request
        logger.info(
            f"Matching request - Title: {title}, Category: {category_slug}, Image: {image_path}, Use Images: {use_images}")

        # Perform matching
        try:
            results = matcher_instance.find_matches(
                title=title,
                image_path_or_url=image_path,
                category_name=category_slug,
                top_n=top_n
            )

            # Add success flag
            results['success'] = True

            # IMPORTANT: Ensure input_item is in results with accessible image path
            if 'input_item' not in results:
                results['input_item'] = {}

            results['input_item'].update({
                'title': title,
                'image_path': web_accessible_path if web_accessible_path else image_path,  # Use web path
                'category': category
            })

            # Store the temp file path in results for LLM reranking
            if temp_file_path:
                results['temp_image_path'] = temp_file_path

            # Log results summary
            logger.info(f"Matching completed - Found {len(results.get('matches', []))} matches")

            return jsonify(results)

        except Exception as e:
            logger.error(f"Error during matching: {str(e)}")
            return jsonify({'error': f'Matching failed: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Error in match endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


# LLM re-ranking endpoint
@app.route('/api/single-item/rerank', methods=['POST'])
def rerank_with_llm():
    """Re-rank matches using LLM"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        matches = data.get('matches', [])
        user_item = data.get('userItem', {})
        category = data.get('category', '')
        temp_image_path = data.get('tempImagePath', '')

        if not matches:
            return jsonify({'error': 'No matches to re-rank'}), 400

        # If temp image path was provided, update the user item
        if temp_image_path and os.path.exists(temp_image_path):
            user_item['image_path'] = temp_image_path
            logger.info(f"Using temp image path for LLM reranking: {temp_image_path}")

        # Import the LLM reranker
        try:
            from SingleItemRerankerLLM import SingleItemLLMReranker

            # Initialize reranker
            reranker = SingleItemLLMReranker(category_name=category)

            # Load user item
            if not reranker.load_user_item(user_item):
                logger.warning("Failed to load user item image for LLM reranking")

            # Re-rank matches
            enhanced_matches = reranker.rerank_matches(
                matches,
                max_to_process=min(20, len(matches)),  # Process top 20 or all if less
                batch_size=5
            )

            # Clean up temp file after LLM processing
            if temp_image_path and temp_image_path in temp_files_to_cleanup:
                try:
                    os.remove(temp_image_path)
                    temp_files_to_cleanup.remove(temp_image_path)
                    logger.info(f"Cleaned up temp file after LLM reranking: {temp_image_path}")
                except Exception as e:
                    logger.error(f"Error cleaning up temp file: {e}")

            return jsonify({
                'success': True,
                'matches': enhanced_matches
            })

        except ImportError as e:
            logger.error(f"Failed to import LLM reranker: {str(e)}")
            logger.warning("LLM reranker not available, returning original matches")
            return jsonify({
                'success': True,
                'matches': matches,
                'message': 'LLM reranker not available'
            })
        except Exception as e:
            logger.error(f"LLM reranking error: {str(e)}")
            return jsonify({
                'success': True,
                'matches': matches,
                'message': f'LLM reranking failed: {str(e)}'
            })

    except Exception as e:
        logger.error(f"Error in rerank endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/single-item/save-results', methods=['POST'])
def save_results():
    """Save match results to file"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Make sure input_item is preserved in the saved data
        # This ensures the web-accessible image path is saved

        # Generate filename
        metadata = data.get('metadata', {})
        category = metadata.get('category', 'results')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"single_item_matches_{category}_{timestamp}.json"

        # Save to file in the same directory as the Flask app
        filepath = Path('.') / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved results to {filename}")

        return jsonify({
            'success': True,
            'filename': filename,
            'path': str(filepath.absolute())
        })

    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        return jsonify({'error': f'Failed to save results: {str(e)}'}), 500


# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'matcher_initialized': matcher is not None,
        'temp_files_pending': len(temp_files_to_cleanup)
    })


# List available categories
@app.route('/api/categories', methods=['GET'])
def list_categories():
    """List available product categories based on existing data files"""
    categories = set()

    # Look for temu_*_analysis.json and amazon_*_analysis.json files
    for file in Path('.').glob('*_analysis.json'):
        match = re.match(r'(temu|amazon)_(.+?)_analysis\.json', file.name)
        if match:
            category = match.group(2)
            categories.add(category)

    # Convert to list and sort
    category_list = sorted(list(categories))

    # Create display names (convert underscores to spaces and title case)
    categories_data = []
    for cat in category_list:
        display_name = cat.replace('_', ' ').title()
        categories_data.append({
            'value': cat,
            'display': display_name
        })

    logger.info(f"Found {len(categories_data)} categories: {[c['value'] for c in categories_data]}")

    return jsonify({
        'categories': categories_data,
        'total': len(categories_data)
    })


@app.route('/api/categories/<category>/count', methods=['GET'])
def get_category_count(category):
    """Get product count for a specific category"""
    try:
        category_slug = sanitize_filename(category)
        temu_file = Path(f"temu_{category_slug}_analysis.json")
        amazon_file = Path(f"amazon_{category_slug}_analysis.json")

        total_count = 0

        if temu_file.exists():
            with open(temu_file, 'r', encoding='utf-8') as f:
                temu_data = json.load(f)
                temu_products = matcher._extract_products(temu_data, 'temu') if matcher else []
                total_count += len(temu_products)

        if amazon_file.exists():
            with open(amazon_file, 'r', encoding='utf-8') as f:
                amazon_data = json.load(f)
                amazon_products = matcher._extract_products(amazon_data, 'amazon') if matcher else []
                total_count += len(amazon_products)

        return jsonify({
            'category': category,
            'total': total_count
        })

    except Exception as e:
        logger.error(f"Error getting count for {category}: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Error handlers
@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


def cleanup_old_uploads(max_age_hours=24):
    """Remove uploads older than specified hours"""
    try:
        upload_dir = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_dir):
            return

        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        for filename in os.listdir(upload_dir):
            if filename.startswith('.'):  # Skip hidden files
                continue

            filepath = os.path.join(upload_dir, filename)
            if os.path.isfile(filepath):
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                if file_time < cutoff_time:
                    try:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old upload: {filename}")
                    except Exception as e:
                        logger.error(f"Error removing {filename}: {e}")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


# Run cleanup occasionally before requests
@app.before_request
def before_request_cleanup():
    """Run cleanup occasionally before requests"""
    import random
    if random.random() < 0.01:  # 1% chance
        cleanup_old_uploads()


# Main entry point
if __name__ == '__main__':
    print("Starting Flask server for Single Item Matcher...")
    print("=" * 50)
    print("Server will be available at: http://localhost:5000")
    print("Dashboard will be at: http://localhost:5000/")
    print("=" * 50)

    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Run the Flask app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,
        debug=True,  # Enable debug mode for development
        threaded=True  # Enable threading for better performance
    )
