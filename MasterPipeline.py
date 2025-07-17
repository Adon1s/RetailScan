"""
Master Pipeline Script for Temu-Amazon Product Matching
Runs all scripts in the correct order and opens the dashboard
Enhanced with comprehensive incremental matching
"""
import subprocess
import sys
import textwrap
import time
import webbrowser
from pathlib import Path
from datetime import datetime
import json
import argparse
import hashlib
from typing import Dict, List, Set, Tuple, Optional


class IncrementalMatchingPipeline:
    """Handles incremental matching logic and comparison tracking"""

    def __init__(self, log_func=print):
        self.log = log_func
        self.comparison_history_file = Path("comparison_history.json")
        self.product_hashes_file = Path("product_hashes.json")
        self.matching_results_file = Path("matching_results.json")

        # Load existing data
        self.comparison_history = self._load_comparison_history()
        self.product_hashes = self._load_product_hashes()
        self.existing_matches = self._load_existing_matches()

    def _load_comparison_history(self) -> Dict:
        """Load history of all comparisons made"""
        if self.comparison_history_file.exists():
            try:
                with open(self.comparison_history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Validate structure
                if "comparisons" in data and "metadata" in data:
                    return data
                else:
                    self.log("Invalid comparison history format, starting fresh", "WARNING")
            except json.JSONDecodeError as e:
                self.log(f"Corrupted comparison history, starting fresh: {e}", "WARNING")

        return {"comparisons": {}, "metadata": {"last_updated": None}}

    def _load_product_hashes(self) -> Dict:
        """Load hashes of products to detect changes"""
        if self.product_hashes_file.exists():
            try:
                with open(self.product_hashes_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                self.log(f"Corrupted product hashes, starting fresh: {e}", "WARNING")
        return {"temu": {}, "amazon": {}}

    def _load_existing_matches(self) -> Dict:
        """Load existing matching results"""
        if self.matching_results_file.exists():
            try:
                with open(self.matching_results_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                self.log(f"Corrupted matching results, starting fresh: {e}", "WARNING")
        return {"metadata": {}, "matches": []}

    def _compute_product_hash(self, product: Dict, platform: str) -> str:
        """Compute hash of product to detect changes"""
        # Include key fields that would affect matching
        if platform == "temu":
            key_fields = f"{product.get('title', '')}{product.get('price', '')}"
        else:  # amazon
            key_fields = f"{product.get('title', '')}{product.get('price', '')}"

        return hashlib.md5(key_fields.encode()).hexdigest()

    def _get_comparison_key(self, temu_id: str, amazon_id: str) -> str:
        """Generate unique key for a comparison"""
        return f"{temu_id}_{amazon_id}"

    def identify_products_to_match(self, temu_products: List[Dict], amazon_products: List[Dict]) -> Dict:
        """Identify which products need matching"""
        results = {
            "new_temu": [],
            "new_amazon": [],
            "updated_temu": [],
            "updated_amazon": [],
            "temu_to_match": [],  # Final list of Temu products to match
            "amazon_to_match": [],  # Final list of Amazon products to match
            "skip_comparisons": set()  # Comparison keys to skip
        }

        # Check Temu products
        for product in temu_products:
            temu_id = product['temu_id']
            current_hash = self._compute_product_hash(product, 'temu')

            if temu_id not in self.product_hashes.get('temu', {}):
                results['new_temu'].append(product)
                results['temu_to_match'].append(product)
            elif self.product_hashes['temu'][temu_id] != current_hash:
                results['updated_temu'].append(product)
                results['temu_to_match'].append(product)
                # Clear comparison history for updated products
                self._clear_comparisons_for_product(temu_id, 'temu')

        # Check Amazon products
        for product in amazon_products:
            amazon_id = product['amazon_id']
            current_hash = self._compute_product_hash(product, 'amazon')

            if amazon_id not in self.product_hashes.get('amazon', {}):
                results['new_amazon'].append(product)
                results['amazon_to_match'].append(product)
            elif self.product_hashes['amazon'][amazon_id] != current_hash:
                results['updated_amazon'].append(product)
                results['amazon_to_match'].append(product)
                # Clear comparison history for updated products
                self._clear_comparisons_for_product(amazon_id, 'amazon')

        # If no new/updated products, use all for consistency
        if not results['temu_to_match']:
            results['temu_to_match'] = temu_products
        if not results['amazon_to_match']:
            results['amazon_to_match'] = amazon_products

        # Identify comparisons to skip (already done and products unchanged)
        for comp_key, comp_data in self.comparison_history.get('comparisons', {}).items():
            temu_id, amazon_id = comp_key.split('_', 1)

            # Skip if both products are unchanged
            temu_unchanged = (temu_id in self.product_hashes.get('temu', {}) and
                            temu_id not in [p['temu_id'] for p in results['temu_to_match']])
            amazon_unchanged = (amazon_id in self.product_hashes.get('amazon', {}) and
                              amazon_id not in [p['amazon_id'] for p in results['amazon_to_match']])

            if temu_unchanged and amazon_unchanged:
                results['skip_comparisons'].add(comp_key)

        # Log summary
        self.log(f"New Temu products: {len(results['new_temu'])}")
        self.log(f"Updated Temu products: {len(results['updated_temu'])}")
        self.log(f"New Amazon products: {len(results['new_amazon'])}")
        self.log(f"Updated Amazon products: {len(results['updated_amazon'])}")
        self.log(f"Comparisons to skip: {len(results['skip_comparisons'])}")

        return results

    def _clear_comparisons_for_product(self, product_id: str, platform: str):
        """Clear comparison history for a specific product"""
        comparisons = self.comparison_history.get('comparisons', {})
        keys_to_remove = []

        for comp_key in comparisons:
            if platform == 'temu' and comp_key.startswith(f"{product_id}_"):
                keys_to_remove.append(comp_key)
            elif platform == 'amazon' and comp_key.endswith(f"_{product_id}"):
                keys_to_remove.append(comp_key)

        for key in keys_to_remove:
            del comparisons[key]

    def update_product_hashes(self, temu_products: List[Dict], amazon_products: List[Dict]):
        """Update stored product hashes"""
        # Update Temu hashes
        for product in temu_products:
            temu_id = product['temu_id']
            self.product_hashes.setdefault('temu', {})[temu_id] = self._compute_product_hash(product, 'temu')

        # Update Amazon hashes
        for product in amazon_products:
            amazon_id = product['amazon_id']
            self.product_hashes.setdefault('amazon', {})[amazon_id] = self._compute_product_hash(product, 'amazon')

        # Save updated hashes
        with open(self.product_hashes_file, 'w', encoding='utf-8') as f:
            json.dump(self.product_hashes, f, indent=2)

    def filter_and_prepare_data(self, matching_info: Dict) -> Tuple[Dict, Dict]:
        """Prepare filtered data for matching"""
        # Create filtered data structures matching the original JSON format
        filtered_temu = {
            "all_products": matching_info['temu_to_match'],
            "metadata": {"filtered": True, "total": len(matching_info['temu_to_match'])}
        }

        filtered_amazon = {
            "all_products": matching_info['amazon_to_match'],
            "metadata": {"filtered": True, "total": len(matching_info['amazon_to_match'])}
        }

        return filtered_temu, filtered_amazon

    def save_comparison_history(self, new_comparisons: List[Dict]):
        """Update comparison history with new comparisons"""
        for match in new_comparisons:
            comp_key = self._get_comparison_key(match['temu_id'], match['amazon_id'])

            self.comparison_history['comparisons'][comp_key] = {
                "date": datetime.now().isoformat(),
                "confidence": match['confidence'],
                "matched": match['confidence'] >= 0.6,  # Your threshold
                "scores": match.get('scores', {})
            }

        self.comparison_history['metadata']['last_updated'] = datetime.now().isoformat()

        with open(self.comparison_history_file, 'w', encoding='utf-8') as f:
            json.dump(self.comparison_history, f, indent=2)

    def merge_results(self, new_results: Dict) -> Dict:
        """Merge new results with existing ones, handling deduplication"""
        # Create a mapping of existing matches for deduplication
        existing_pairs = {(m['temu_id'], m['amazon_id']): m for m in self.existing_matches.get('matches', [])}

        # Process new matches
        merged_matches = []
        updated_count = 0
        new_count = 0

        for new_match in new_results.get('matches', []):
            pair_key = (new_match['temu_id'], new_match['amazon_id'])

            if pair_key in existing_pairs:
                # Update existing match (might have new scores)
                existing = existing_pairs[pair_key]
                if existing['confidence'] != new_match['confidence']:
                    merged_matches.append(new_match)
                    updated_count += 1
                else:
                    merged_matches.append(existing)
            else:
                # New match
                merged_matches.append(new_match)
                new_count += 1
                existing_pairs[pair_key] = new_match

        # Add remaining existing matches that weren't updated
        for pair_key, match in existing_pairs.items():
            if not any(m['temu_id'] == pair_key[0] and m['amazon_id'] == pair_key[1] for m in merged_matches):
                merged_matches.append(match)

        # Sort by confidence
        merged_matches.sort(key=lambda x: x['confidence'], reverse=True)

        # Update metadata
        metadata = new_results.get('metadata', {}).copy()
        metadata.update({
            'total_matches': len(merged_matches),
            'high_confidence': len([m for m in merged_matches if m['confidence'] >= 0.8]),
            'medium_confidence': len([m for m in merged_matches if 0.6 <= m['confidence'] < 0.8]),
            'low_confidence': len([m for m in merged_matches if m['confidence'] < 0.6]),
            'needs_review': len([m for m in merged_matches if m.get('needs_review', False)]),
            'new_matches': new_count,
            'updated_matches': updated_count,
            'merge_timestamp': datetime.now().isoformat()
        })

        self.log(f"Merged results: {new_count} new, {updated_count} updated, {len(merged_matches)} total matches")

        return {'metadata': metadata, 'matches': merged_matches}


class PipelineRunner:
    def __init__(self, skip_scraping=False, skip_llm=False, verbose=True, use_http_server=False):
        self.skip_scraping = skip_scraping
        self.skip_llm = skip_llm
        self.verbose = verbose
        self.use_http_server = use_http_server
        self.start_time = time.time()
        self.log_dir = Path("pipeline_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f"pipeline_run_{timestamp}.log"

        # Initialize incremental matching handler
        self.incremental_matcher = IncrementalMatchingPipeline(log_func=self.log)

    def log(self, message, level="INFO"):
        """Log message to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"

        if self.verbose or level in ["ERROR", "WARNING"]:
            print(log_message)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

    def run_script(self, script_name: str, description: str) -> bool:
        """Run a Python script and capture output"""
        self.log(f"Running: {description}")
        try:
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )

            if result.returncode == 0:
                self.log(f"✓ {description} completed successfully")
                if result.stderr and self.verbose:
                    self.log(f"Script output (stderr): {result.stderr}")
                return True
            else:
                self.log(f"✗ {description} failed with return code {result.returncode}", "ERROR")
                if result.stderr:
                    self.log(f"Error output: {result.stderr}", "ERROR")
                return False
        except Exception as e:
            self.log(f"✗ Failed to run {script_name}: {str(e)}", "ERROR")
            return False

    def run_matcher_with_filtered_data(self, filtered_temu: Dict, filtered_amazon: Dict) -> bool:
        """Run the matcher with pre-filtered data"""
        # Save filtered data to temporary files
        temp_temu = Path("temp_filtered_temu.json")
        temp_amazon = Path("temp_filtered_amazon.json")
        wrapper_file = Path("temp_matcher_wrapper.py")

        try:
            with open(temp_temu, 'w', encoding='utf-8') as f:
                json.dump(filtered_temu, f, indent=2)

            with open(temp_amazon, 'w', encoding='utf-8') as f:
                json.dump(filtered_amazon, f, indent=2)

            # Create a modified version of the matcher that uses these files
            wrapper_code = textwrap.dedent(f'''
                import sys
                import json
                from pathlib import Path

                sys.path.insert(0, ".")

                try:
                    from TemuAmazonProductMatcher import ProductMatcher

                    # Initialize matcher
                    matcher = ProductMatcher(use_images=True)

                    # Run matching with filtered files
                    results = matcher.match_all_products(
                        "{temp_temu}",
                        "{temp_amazon}"
                    )

                    # Save results
                    matcher.save_results(results, "temp_matching_results.json")

                    print("Matching completed successfully")
                    sys.exit(0)

                except Exception as e:
                    print(f"Error in matcher wrapper: {{e}}", file=sys.stderr)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    sys.exit(1)
            ''')

            with open(wrapper_file, 'w', encoding='utf-8') as f:
                f.write(wrapper_code)

            # Run the wrapper
            success = self.run_script(str(wrapper_file), "Running filtered matching")

            return success

        except Exception as e:
            self.log(f"Error in run_matcher_with_filtered_data: {e}", "ERROR")
            return False

        finally:
            # Cleanup temp files (but NOT the results file - that's handled by caller)
            for temp_file in [temp_temu, temp_amazon, wrapper_file]:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except Exception as e:
                        self.log(f"Warning: Could not delete {temp_file}: {e}", "WARNING")

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

        # Clean up any leftover temp files from previous failed runs
        temp_results_file = Path("temp_matching_results.json")
        if temp_results_file.exists():
            self.log("Cleaning up leftover temp file from previous run")
            try:
                temp_results_file.unlink()
            except Exception as e:
                self.log(f"Warning: Could not clean up temp file: {e}", "WARNING")

        # Check required files
        if not self.check_required_files():
            self.log("Aborting: Missing required files", "ERROR")
            return False

        # Step 1 & 2: Scraping (can be run in parallel)
        if not self.skip_scraping:
            self.log("\n--- STEP 1 & 2: Web Scraping ---")

            if not self.run_script("TemuScraper.py", "Scraping Temu products"):
                self.log("Temu scraping failed. Continue anyway? (y/n): ", "WARNING")
                if input().lower() != 'y':
                    return False

            if not self.run_script("AmazonScraper.py", "Scraping Amazon products"):
                self.log("Amazon scraping failed. Continue anyway? (y/n): ", "WARNING")
                if input().lower() != 'y':
                    return False

            self.check_output_files(["temu_baby_toys.csv", "amazon_baby_toys.csv"], "Scraping")
        else:
            self.log("\n--- Skipping scraping (using existing CSVs) ---")

        # Step 3 & 4: Data Organization
        self.log("\n--- STEP 3 & 4: Data Organization ---")

        if not self.run_script("TemuDataOrganizer.py", "Organizing Temu data"):
            return False

        if not self.run_script("AmazonDataOrganizer.py", "Organizing Amazon data"):
            return False

        if not self.check_output_files(
                ["temu_products_for_analysis.json", "amazon_products_for_analysis.json"],
                "Data organization"
        ):
            return False

        # Step 5: Incremental Product Matching
        self.log("\n--- STEP 5: Incremental Product Matching ---")

        # Load organized data
        try:
            with open("temu_products_for_analysis.json", 'r', encoding='utf-8') as f:
                temu_data = json.load(f)
            temu_products = temu_data.get('all_products', [])

            with open("amazon_products_for_analysis.json", 'r', encoding='utf-8') as f:
                amazon_data = json.load(f)
            amazon_products = amazon_data.get('all_products', [])
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.log(f"Error loading product data: {e}", "ERROR")
            return False

        # Identify what needs matching
        matching_info = self.incremental_matcher.identify_products_to_match(temu_products, amazon_products)

        # Check if any matching is needed
        total_new = (len(matching_info['new_temu']) + len(matching_info['new_amazon']) +
                     len(matching_info['updated_temu']) + len(matching_info['updated_amazon']))

        if total_new == 0 and self.incremental_matcher.existing_matches.get('matches'):
            self.log("No new or updated products found. Using existing matches.")
            # Still update hashes for consistency
            self.incremental_matcher.update_product_hashes(temu_products, amazon_products)
        else:
            # Prepare filtered data
            filtered_temu, filtered_amazon = self.incremental_matcher.filter_and_prepare_data(matching_info)

            try:
                # Run matching on filtered data
                if not self.run_matcher_with_filtered_data(filtered_temu, filtered_amazon):
                    self.log("Matching failed", "ERROR")
                    return False

                # Verify temp file was created
                if not temp_results_file.exists():
                    self.log("Matching failed to create results file", "ERROR")
                    return False

                # Load new results
                try:
                    with open(temp_results_file, 'r', encoding='utf-8') as f:
                        new_results = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    self.log(f"Error loading temp results: {e}", "ERROR")
                    return False

                # Update comparison history
                self.incremental_matcher.save_comparison_history(new_results.get('matches', []))

                # Merge with existing results
                merged_results = self.incremental_matcher.merge_results(new_results)

                # Save merged results
                try:
                    with open("matching_results.json", 'w', encoding='utf-8') as f:
                        json.dump(merged_results, f, indent=2)
                except Exception as e:
                    self.log(f"Error saving merged results: {e}", "ERROR")
                    return False

                # Update product hashes
                self.incremental_matcher.update_product_hashes(temu_products, amazon_products)

                self.log("Incremental matching completed successfully")

            finally:
                # Always clean up temp file, even if something failed
                if temp_results_file.exists():
                    try:
                        temp_results_file.unlink()
                    except Exception as e:
                        self.log(f"Warning: Could not delete temp results file: {e}", "WARNING")

        # Check for matching results
        if not self.check_output_files(["matching_results.json"], "Product matching"):
            return False

        # Step 6: LLM Enhancement (optional)
        if not self.skip_llm:
            self.log("\n--- STEP 6: LLM Enhancement ---")

            # Only run LLM on new/updated matches if needed
            try:
                with open("matching_results.json", 'r', encoding='utf-8') as f:
                    current_results = json.load(f)

                new_match_count = current_results.get('metadata', {}).get('new_matches', 0)
                if new_match_count > 0:
                    self.log(f"Running LLM analysis on {new_match_count} new matches")
                    if not self.run_script(
                            "ImageFinalPassLLM.py",
                            "Running LLM re-ranking on matches"):
                        self.log("LLM enhancement failed – continuing pipeline", "WARNING")
                else:
                        self.log("No new matches found—skipping LLM enhancement")
            except Exception as e:
                self.log(f"Error in LLM enhancement: {e}", "WARNING")
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

            try:
                webbrowser.open(file_url)

                # Show instructions for HTTP server if needed
                self.log("\nDashboard opened in browser.")
                self.log("If you see CORS errors or need HTTP features:")
                self.log("  Run: python run_pipeline.py --step serve")
                self.log("  Or: python -m http.server 8000")
            except Exception as e:
                self.log(f"Error opening dashboard: {e}", "ERROR")

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
            # Try to load LLM analyzed results first
            results_file = "matching_results_llm.json"
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

            # Show incremental stats if available
            if 'new_matches' in metadata:
                self.log(f"\nIncremental matching stats:")
                self.log(f"  New matches: {metadata.get('new_matches', 0)}")
                self.log(f"  Updated matches: {metadata.get('updated_matches', 0)}")

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
    parser.add_argument("--clear-history", action="store_true",
                        help="Clear comparison history and start fresh")

    args = parser.parse_args()

    # Handle clearing history
    if args.clear_history:
        print("Clearing comparison history...")
        for file in ["comparison_history.json", "product_hashes.json"]:
            if Path(file).exists():
                Path(file).unlink()
                print(f"  Deleted {file}")
        print("History cleared. Run pipeline again to start fresh.")
        return

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
            runner.use_http_server = True
            runner.serve_dashboard_with_http()
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
            use_http_server=True
        )
        runner.run_pipeline()


if __name__ == "__main__":
    main()
