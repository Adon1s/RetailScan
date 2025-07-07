"""
Master Pipeline Script for Temu-Amazon Product Matching
Runs all scripts in the correct order and opens the dashboard
"""
import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path
from datetime import datetime
import json
import argparse


class PipelineRunner:

    def __init__(self, skip_scraping=False, skip_llm=False, verbose=True, use_http_server=False):
        self.skip_scraping = skip_scraping
        self.skip_llm = skip_llm
        self.verbose = verbose
        self.use_http_server = use_http_server
        self.start_time = time.time()
        # ensure logs directory exists
        self.log_dir = Path("pipeline_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # log into pipeline_logs/
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f"pipeline_run_{timestamp}.log"

    def log(self, message, level="INFO"):
        """Log message to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"

        if self.verbose or level in ["ERROR", "WARNING"]:
            print(log_message)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

    def run_script(self, script_name, description, timeout=None):
        """Run a Python script and stream its output directly to the terminal"""
        self.log(f"Starting: {description}")
        self.log(f"Running: python {script_name}")

        try:
            # run the child process un-buffered; inherit our stdout/stderr
            result = subprocess.run(
                [sys.executable, "-u", script_name],  # <- -u = unbuffered
                check=False,  # don't raise on non-zero
                timeout=timeout  # still honours any timeout
            )

            if result.returncode != 0:
                self.log(f"{script_name} exited with code {result.returncode}", "ERROR")
                return False

            self.log(f"✓ Completed: {description}")
            return True

        except subprocess.TimeoutExpired:
            self.log(f"Timeout running {script_name} after {timeout} seconds", "ERROR")
            return False
        except Exception as e:
            self.log(f"Exception running {script_name}: {e}", "ERROR")
            return False

    def check_required_files(self):
        """Check if all required scripts exist"""
        required_scripts = [
            "TemuScraper.py",
            "AmazonScraper.py",
            "AmazonDataOrganizer.py",
            "TemuDataOrganizer.py",
            "TemuAmazonProductMatcher.py",
            "ImageFinalPassLLM.py"
        ]

        if not self.skip_scraping:
            scripts_to_check = required_scripts
        else:
            scripts_to_check = required_scripts[2:]  # Skip scraper checks

        if self.skip_llm:
            scripts_to_check = [s for s in scripts_to_check if "LLM" not in s]

        missing = []
        for script in scripts_to_check:
            if not Path(script).exists():
                missing.append(script)

        if missing:
            self.log(f"Missing required scripts: {', '.join(missing)}", "ERROR")
            return False

        # Check for dashboard
        if not Path("ProductMatchingDashboard.html").exists():
            self.log("Warning: ProductMatchingDashboard.html not found", "WARNING")

        return True

    def check_output_files(self, files, step_name):
        """Check if expected output files were created"""
        missing = []
        for file in files:
            if not Path(file).exists():
                missing.append(file)

        if missing:
            self.log(f"{step_name} may have failed. Missing files: {', '.join(missing)}", "WARNING")
            return False
        return True

    def run_pipeline(self):
        """Run the complete pipeline"""
        self.log("=" * 60)
        self.log("Starting Temu-Amazon Product Matching Pipeline")
        self.log("=" * 60)

        # Check required files
        if not self.check_required_files():
            self.log("Aborting: Missing required files", "ERROR")
            return False

        # Step 1 & 2: Scraping (can be run in parallel)
        if not self.skip_scraping:
            self.log("\n--- STEP 1 & 2: Web Scraping ---")

            # Run scrapers
            if not self.run_script("TemuScraper.py", "Scraping Temu products"):
                self.log("Temu scraping failed. Continue anyway? (y/n): ", "WARNING")
                if input().lower() != 'y':
                    return False

            if not self.run_script("AmazonScraper.py", "Scraping Amazon products"):
                self.log("Amazon scraping failed. Continue anyway? (y/n): ", "WARNING")
                if input().lower() != 'y':
                    return False

            # Check for CSV outputs
            self.check_output_files(["temu_baby_toys.csv", "amazon_baby_toys.csv"], "Scraping")
        else:
            self.log("\n--- Skipping scraping (using existing CSVs) ---")

        # Step 3 & 4: Data Organization
        self.log("\n--- STEP 3 & 4: Data Organization ---")

        if not self.run_script("TemuDataOrganizer.py", "Organizing Temu data"):
            return False

        if not self.run_script("AmazonDataOrganizer.py", "Organizing Amazon data"):
            return False

        # Check for JSON outputs
        if not self.check_output_files(
                ["temu_products_for_analysis.json", "amazon_products_for_analysis.json"],
                "Data organization"
        ):
            return False

        # Step 5: Product Matching
        self.log("\n--- STEP 5: Product Matching ---")

        if not self.run_script("TemuAmazonProductMatcher.py", "Matching products between platforms"):
            return False

        # Check for matching results
        if not self.check_output_files(["matching_results.json"], "Product matching"):
            return False

        # Step 6: LLM Enhancement (optional)
        if not self.skip_llm:
            self.log("\n--- STEP 6: LLM Enhancement ---")

            # non-interactive: just run it
            if not self.run_script(
                    "ImageFinalPassLLM.py",
                    "Running LLM re-ranking on top matches"):
                self.log("LLM enhancement failed – continuing pipeline", "WARNING")
        else:
            self.log("\n--- Skipping LLM enhancement ---")

        # Step 7: Open Dashboard
        self.log("\n--- STEP 7: Opening Dashboard ---")

        dashboard_path = Path("ProductMatchingDashboard.html")
        if not dashboard_path.exists():
            self.log("Dashboard file not found.", "ERROR")
            return False

        if self.use_http_server:
            # Option: Use HTTP server (for CORS or API needs)
            self.serve_dashboard_with_http()
        else:
            # Default: Use file:// protocol (simpler and more reliable)
            file_url = dashboard_path.absolute().as_uri()
            self.log(f"Opening dashboard: {file_url}")
            webbrowser.open(file_url)

            # Show instructions for HTTP server if needed
            self.log("\nDashboard opened in browser.")
            self.log("If you see CORS errors or need HTTP features:")
            self.log("  Run: python run_pipeline.py --step serve")
            self.log("  Or: python -m http.server 8000")

        # Summary
        elapsed_time = time.time() - self.start_time
        self.log("\n" + "=" * 60)
        self.log(f"Pipeline completed in {elapsed_time / 60:.1f} minutes")
        self.log(f"Log saved to: {self.log_file}")

        # Show final statistics
        self.show_summary()

        return True

    def serve_dashboard_with_http(self):
        """Serve dashboard with HTTP server"""
        try:
            from http.server import HTTPServer, SimpleHTTPRequestHandler
            import threading
            import socket

            # Get free port
            with socket.socket() as s:
                s.bind(('', 0))
                port = s.getsockname()[1]

            # Start server in background
            server = HTTPServer(('localhost', port), SimpleHTTPRequestHandler)
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.daemon = True
            server_thread.start()

            # Wait for server to start
            time.sleep(0.5)

            # Open browser
            url = f"http://localhost:{port}/ProductMatchingDashboard.html"
            self.log(f"Opening dashboard at: {url}")
            webbrowser.open(url)

            # Keep server running
            self.log("\nDashboard server running. Press Enter to exit...")
            try:
                input()
            except KeyboardInterrupt:
                pass

            self.log("Shutting down server...")
            server.shutdown()

        except Exception as e:
            self.log(f"Failed to start HTTP server: {e}", "ERROR")
            self.log("Falling back to file:// protocol")
            file_url = Path("ProductMatchingDashboard.html").absolute().as_uri()
            webbrowser.open(file_url)

    def run_http_server(self, port=8000):
        """Run a dedicated HTTP server"""
        import http.server
        import socketserver

        self.log(f"Starting HTTP server on port {port}...")

        Handler = http.server.SimpleHTTPRequestHandler

        try:
            with socketserver.TCPServer(("", port), Handler) as httpd:
                self.log(f"Server running at http://localhost:{port}/")
                self.log(f"Open: http://localhost:{port}/ProductMatchingDashboard.html")
                self.log("Press Ctrl+C to stop the server")
                httpd.serve_forever()
        except KeyboardInterrupt:
            self.log("\nServer stopped.")
        except OSError as e:
            if e.errno == 48:  # Port already in use
                self.log(f"Port {port} is already in use. Try a different port:", "ERROR")
                self.log(f"  python run_pipeline.py --step serve --port {port + 1}")
            else:
                self.log(f"Server error: {e}", "ERROR")

    def show_summary(self):
        """Show summary of results"""
        try:
            # Try to load enriched results first
            results_file = "matching_results_enriched.json"
            if not Path(results_file).exists():
                results_file = "matching_results.json"

            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            metadata = data.get('metadata', {})
            self.log("\n--- Results Summary ---")
            self.log(f"Total matches found: {metadata.get('total_matches', 0)}")
            self.log(f"High confidence (≥0.8): {metadata.get('high_confidence', 0)}")
            self.log(f"Medium confidence (0.6-0.8): {metadata.get('medium_confidence', 0)}")
            self.log(f"Low confidence (<0.6): {metadata.get('low_confidence', 0)}")

            if 'llm_enrichment' in metadata:
                llm_data = metadata['llm_enrichment']
                self.log(f"\nLLM Enhancement:")
                self.log(f"  Processed: {llm_data.get('total_processed', 0)}")
                self.log(f"  Same product: {llm_data.get('llm_same', 0)}")
                self.log(f"  Different product: {llm_data.get('llm_different', 0)}")

        except Exception as e:
            self.log(f"Could not load results summary: {e}", "WARNING")


def main():
    parser = argparse.ArgumentParser(description="Run the complete Temu-Amazon matching pipeline")
    parser.add_argument("--skip-scraping", action="store_true",
                        help="Skip web scraping (use existing CSV files)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM enhancement step")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    parser.add_argument("--http-server", action="store_true",
                        help="Use HTTP server for dashboard instead of file://")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for HTTP server (default: 8000)")
    parser.add_argument("--step", type=str,
                        help="Run only a specific step: scrape, organize, match, llm, dashboard, serve")

    args = parser.parse_args()

    # Handle single step execution
    if args.step:
        runner = PipelineRunner(verbose=not args.quiet)

        if args.step == "scrape":
            runner.run_script("TemuScraper.py", "Scraping Temu products")
            runner.run_script("AmazonScraper.py", "Scraping Amazon products")
        elif args.step == "organize":
            runner.run_script("TemuDataOrganizer.py", "Organizing Temu data")
            runner.run_script("AmazonDataOrganizer.py", "Organizing Amazon data")
        elif args.step == "match":
            runner.run_script("TemuAmazonProductMatcher.py", "Matching products")
        elif args.step == "llm":
            runner.run_script("ImageFinalPassLLM.py", "LLM enhancement")
        elif args.step == "dashboard":
            dashboard_path = Path("ProductMatchingDashboard.html").absolute()
            if args.http_server:
                runner.use_http_server = True
                runner.serve_dashboard_with_http()
            else:
                webbrowser.open(f"file://{dashboard_path}")
        elif args.step == "serve":
            # Run dedicated HTTP server
            runner.run_http_server(port=args.port)
        else:
            print(f"Unknown step: {args.step}")
            print("Valid steps: scrape, organize, match, llm, dashboard, serve")
            return
    else:
        # Run full pipeline
        runner = PipelineRunner(
            skip_scraping=args.skip_scraping,
            skip_llm=args.skip_llm,
            verbose=not args.quiet,
            use_http_server=args.http_server
        )
        runner.run_pipeline()


if __name__ == "__main__":
    main()
