"""
Enhanced Pipeline API Server with Remote Access Support
"""
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
import subprocess
import threading
import queue
import json
import sys
import os
import time
import socket
from pathlib import Path
import re
from datetime import datetime
import argparse

app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for all routes

# Global state
pipeline_process = None
pipeline_running = False
message_queue = queue.Queue()
pipeline_thread = None

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

def sanitize_filename(filename: str) -> str:
    """Sanitize string for use in filenames"""
    sanitized = re.sub(r'[^\w\s-]', '', filename)
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    return sanitized.lower()

def normalize_search_terms(search_terms: str) -> str:
    """Normalize search terms for display/searching"""
    cleaned = re.sub(r'[^\w\s_-]', '', search_terms.strip())
    normalized = re.sub(r'[_-]+', ' ', cleaned)
    return ' '.join(normalized.split())

def stream_pipeline_output(process, search_terms):
    """Stream pipeline output to message queue"""
    global pipeline_running

    try:
        # Send initial message
        message_queue.put(json.dumps({
            'type': 'log',
            'message': f'Starting pipeline for "{search_terms}"...'
        }))

        # Stream stdout
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            line = line.strip()
            if line:
                message_queue.put(json.dumps({
                    'type': 'log',
                    'message': line
                }))

        # Wait for process to complete
        process.wait()

        # Send completion message
        if process.returncode == 0:
            # Try to load results
            search_slug = sanitize_filename(search_terms)
            results_file = f"matching_results_{search_slug}.json"
            enriched_file = f"matching_results_enriched_{search_slug}.json"

            results = {}
            try:
                # Try enriched file first
                if Path(enriched_file).exists():
                    with open(enriched_file, 'r') as f:
                        data = json.load(f)
                elif Path(results_file).exists():
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                else:
                    data = {}

                metadata = data.get('metadata', {})
                results = {
                    'totalMatches': metadata.get('total_matches', 0),
                    'highConfidence': metadata.get('high_confidence', 0),
                    'llmProcessed': metadata.get('llm_enrichment', {}).get('total_processed', 0)
                }
            except Exception as e:
                print(f"Error loading results: {e}")

            # Send completion with dashboard URL
            message_queue.put(json.dumps({
                'type': 'complete',
                'results': results,
                'dashboardUrl': f'/ProductMatchingDashboard.html?category={search_slug}'
            }))
        else:
            message_queue.put(json.dumps({
                'type': 'error',
                'message': f'Pipeline failed with code {process.returncode}'
            }))

    except Exception as e:
        message_queue.put(json.dumps({
            'type': 'error',
            'message': str(e)
        }))

    finally:
        pipeline_running = False

@app.route('/')
def index():
    """Serve the main dashboard"""
    return send_file('PipelineDashboard.html')

@app.route('/ProductMatchingDashboard.html')
def serve_product_dashboard():
    """Serve the product matching dashboard"""
    return send_file('ProductMatchingDashboard.html')

@app.route('/SingleItemResultsDashboard.html')
def serve_single_item_dashboard():
    """Serve the single item results dashboard"""
    if Path('SingleItemResultsDashboard.html').exists():
        return send_file('SingleItemResultsDashboard.html')
    return "Single Item Results Dashboard not found", 404

@app.route('/api/status')
def status():
    """Get server status and connection info"""
    local_ip = get_local_ip()
    return jsonify({
        'status': 'running',
        'pipeline_running': pipeline_running,
        'local_ip': local_ip,
        'port': request.environ.get('SERVER_PORT', 5000),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/pipeline/start', methods=['POST'])
def start_pipeline():
    """Start the pipeline process"""
    global pipeline_process, pipeline_running, pipeline_thread

    if pipeline_running:
        return jsonify({'error': 'Pipeline already running'}), 400

    try:
        data = request.json
        search_terms = data.get('searchTerms', 'baby toys')
        skip_scraping = data.get('skipScraping', False)
        skip_llm = data.get('skipLLM', False)
        num_pages = data.get('numPages', 1)

        # Clear message queue
        while not message_queue.empty():
            message_queue.get()

        # Build command - add --no-dashboard flag
        cmd = [sys.executable, 'MasterPipeline.py', '--search', search_terms, '--no-dashboard']

        if skip_scraping:
            cmd.append('--skip-scraping')
        if skip_llm:
            cmd.append('--skip-llm')

        # Prepare input for number of pages
        input_text = f"{num_pages}\n"

        # Start pipeline process
        # Set environment for UTF-8 encoding on Windows
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        pipeline_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            env=env
        )

        # Send input for number of pages
        if not skip_scraping:
            pipeline_process.stdin.write(input_text)
            pipeline_process.stdin.flush()
            pipeline_process.stdin.close()  # Close stdin after sending input

        pipeline_running = True

        # Start thread to stream output
        pipeline_thread = threading.Thread(
            target=stream_pipeline_output,
            args=(pipeline_process, search_terms)
        )
        pipeline_thread.daemon = True
        pipeline_thread.start()

        return jsonify({'status': 'started'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pipeline/stop', methods=['POST'])
def stop_pipeline():
    """Stop the running pipeline"""
    global pipeline_process, pipeline_running

    if not pipeline_running or not pipeline_process:
        return jsonify({'error': 'No pipeline running'}), 400

    try:
        pipeline_process.terminate()
        pipeline_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pipeline_process.kill()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    pipeline_running = False
    message_queue.put(json.dumps({
        'type': 'error',
        'message': 'Pipeline stopped by user'
    }))

    return jsonify({'status': 'stopped'}), 200

@app.route('/api/pipeline/stream')
def stream_events():
    """Server-Sent Events endpoint for real-time updates"""
    def generate():
        while True:
            try:
                # Get message from queue with timeout
                message = message_queue.get(timeout=1)
                yield f"data: {message}\n\n"

                # Check if this was a completion message
                msg_data = json.loads(message)
                if msg_data.get('type') in ['complete', 'error']:
                    break

            except queue.Empty:
                # Send heartbeat to keep connection alive
                if pipeline_running:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                else:
                    # If pipeline is not running and queue is empty, we're done
                    break
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                break

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'  # Disable Nginx buffering
        }
    )

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get available categories from existing files"""
    categories = []

    # Look for matching results files
    for file in Path('.').glob('matching_results_*.json'):
        match = re.match(r'matching_results_(.+)\.json', file.name)
        if match and match.group(1) != 'enriched':
            slug = match.group(1)
            # Convert slug back to display name
            display_name = slug.replace('_', ' ').title()
            categories.append({
                'slug': slug,
                'display': display_name
            })

    return jsonify(categories)

# Serve static files (for all assets)
@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    file_path = Path(path)
    if file_path.exists() and file_path.is_file():
        return send_file(path)
    return "Not found", 404

def main():
    parser = argparse.ArgumentParser(description="Pipeline API Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    local_ip = get_local_ip()

    print("=" * 60)
    print("Starting Pipeline API Server...")
    print("=" * 60)
    print(f"Local access:    http://localhost:{args.port}")
    print(f"Network access:  http://{local_ip}:{args.port}")
    print()
    print("For remote access from another network:")
    print("1. Set up port forwarding on your router")
    print("2. Use a tunneling service like ngrok:")
    print(f"   ngrok http {args.port}")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")

    # Run the Flask server
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )

if __name__ == '__main__':
    main()