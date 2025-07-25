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
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
import re
import atexit
import shutil

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
    if os.path.exists('SingleItemProductMatchingDashboard.html'):
        return send_from_directory('.', 'SingleItemProductMatchingDashboard.html')
    else:
        return "Dashboard HTML file not found. Please ensure SingleItemProductMatchingDashboard.html is in the same directory.", 404


# Serve static files (for images, CSS, JS if needed)
@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    if os.path.exists(filename):
        return send_from_directory('.', filename)
    return "File not found", 404


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

        # Check for uploaded file
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename and allowed_file(file.filename):
                # Save uploaded file temporarily
                filename = secure_filename(file.filename)
                temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{datetime.now().timestamp()}_{filename}")
                file.save(temp_file_path)
                image_path = temp_file_path
                # Track for cleanup later (not immediately)
                temp_files_to_cleanup.add(temp_file_path)
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


# Save results endpoint
@app.route('/api/single-item/save-results', methods=['POST'])
def save_results():
    """Save match results to file"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

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