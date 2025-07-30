"""
Master Pipeline Script for Temu-Amazon Product Matching
Runs all scripts in the correct order and opens the dashboard
Enhanced with comprehensive incremental matching and dynamic search terms
"""
import subprocess
import sys
import textwrap
import threading
import time
import webbrowser
from pathlib import Path
from datetime import datetime
import json
import argparse
import hashlib
import re
from typing import Dict, List, Set, Tuple, Optional


def sanitize_filename(filename: str) -> str:
    """Sanitize string for use in filenames (matches scraper logic)"""
    sanitized = re.sub(r'[^\w\s-]', '', filename)
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    return sanitized.lower()


def normalize_search_terms(search_terms: str) -> str:
    """Normalize search terms for display/searching (convert underscores to spaces)"""
    # First sanitize to remove special chars, then convert underscores back to spaces
    cleaned = re.sub(r'[^\w\s_-]', '', search_terms.strip())
    normalized = re.sub(r'[_-]+', ' ', cleaned)
    return ' '.join(normalized.split())  # Normalize multiple spaces to single space


class IncrementalMatchingPipeline:
    """Handles incremental matching logic and comparison tracking"""

    def __init__(self, log_func=print, category_slug=None):
        self.log = log_func
        self.category_slug = category_slug

        # Use category-specific files if category is provided
        if category_slug:
            self.comparison_history_file = Path(f"comparison_history_{category_slug}.json")
            self.product_hashes_file = Path(f"product_hashes_{category_slug}.json")
            self.matching_results_file = Path(f"matching_results_{category_slug}.json")
        else:
            # Fallback to global files (not recommended)
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
    def __init__(self, skip_scraping=False, skip_llm=False, verbose=True, use_http_server=True, search_terms=None,
                 no_dashboard=False):
        self.skip_scraping = skip_scraping
        self.skip_llm = skip_llm
        self.verbose = verbose
        self.use_http_server = use_http_server
        self.no_dashboard = no_dashboard  # Add this line

        # Normalize search terms - convert underscores to spaces for display/search
        self.search_terms = normalize_search_terms(search_terms or "baby toys")
        # Create sanitized slug for filenames (always uses underscores)
        self.search_slug = sanitize_filename(self.search_terms)

        # Dynamic file paths based on search terms
        self.temu_csv = Path(f"temu_{self.search_slug}.csv")
        self.amazon_csv = Path(f"amazon_{self.search_slug}.csv")
        self.temu_json = Path(f"temu_{self.search_slug}_analysis.json")
        self.amazon_json = Path(f"amazon_{self.search_slug}_analysis.json")

        self.start_time = time.time()
        self.log_dir = Path("pipeline_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f"pipeline_run_{timestamp}.log"

        # Initialize incremental matching handler WITH CATEGORY SLUG
        self.incremental_matcher = IncrementalMatchingPipeline(log_func=self.log, category_slug=self.search_slug)

        # Save config for other scripts to use
        self.save_pipeline_config()

    def save_pipeline_config(self):
        """Save pipeline configuration for other scripts"""
        config = {
            "search_terms": self.search_terms,  # Normalized version with spaces
            "search_slug": self.search_slug,     # Sanitized version with underscores
            "temu_csv": str(self.temu_csv),
            "amazon_csv": str(self.amazon_csv),
            "temu_json": str(self.temu_json),
            "amazon_json": str(self.amazon_json)
        }
        with open("pipeline_config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        self.log(f"Saved pipeline config with search terms: '{self.search_terms}'")

    def log(self, message, level="INFO"):
        """Log message to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"

        if self.verbose or level in ["ERROR", "WARNING"]:
            print(log_message)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

    def find_available_data_files(self):
        """Find available CSV and JSON files for different search terms"""
        available_terms = set()

        # Check for CSV files
        for csv_file in Path(".").glob("temu_*.csv"):
            match = re.match(r"temu_(.+)\.csv", csv_file.name)
            if match:
                term_slug = match.group(1)
                # Check if corresponding Amazon CSV exists
                if Path(f"amazon_{term_slug}.csv").exists():
                    available_terms.add(term_slug)

        # Convert slugs back to readable terms
        return [term.replace('_', ' ') for term in sorted(available_terms)]

    def run_script(self, script_name: str, description: str, input_text: str = None) -> bool:
        """Run a Python script with real-time streaming output"""
        self.log(f"Running: {description}")
        try:
            # Prepare command
            cmd = [sys.executable, script_name]

            # Start subprocess with pipes for streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if input_text else None,
                text=True,
                encoding='utf-8',
                errors='replace'  # Handle any encoding issues gracefully
            )

            # Provide input if needed
            if input_text:
                process.stdin.write(input_text)
                process.stdin.close()

            # Stream output in real-time (non-blocking)
            def stream_output(pipe, level="INFO"):
                """Helper to stream from a pipe"""
                for line in iter(pipe.readline, ''):
                    line = line.strip()
                    if line:
                        print(line)  # Print to console immediately
                        self.log(line, level)  # Also log it

            # Start threads for streaming stdout and stderr
            stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, "INFO"))
            stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, "ERROR"))

            stdout_thread.start()
            stderr_thread.start()

            # Wait for process to finish
            process.wait()

            # Join threads
            stdout_thread.join()
            stderr_thread.join()

            if process.returncode == 0:
                self.log(f"✓ {description} completed successfully")
                return True
            else:
                self.log(f"✗ {description} failed with return code {process.returncode}", "ERROR")
                return False

        except Exception as e:
            self.log(f"✗ Failed to run {script_name}: {str(e)}", "ERROR")
            return False

    def run_scraper_with_config(self, scraper_script: str, platform: str, num_pages: int = 1) -> bool:
        """Run scraper with predefined inputs"""
        # Create wrapper script that provides inputs automatically
        wrapper_file = Path(f"temp_{platform}_scraper_wrapper.py")

        wrapper_code = textwrap.dedent(f'''
            import subprocess
            import sys
            
            # Prepare inputs for the scraper (use normalized search terms with spaces)
            inputs = "{self.search_terms}\\n{num_pages}\\n"
            
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

            success = self.run_script(str(wrapper_file), f"Scraping {platform} products")
            return success

        finally:
            if wrapper_file.exists():
                wrapper_file.unlink()

    def run_scrapers_parallel(self, num_pages: int = 1) -> Tuple[bool, bool]:
        """Run both scrapers in parallel"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        # Thread-safe result storage
        results = {'temu': False, 'amazon': False}
        results_lock = threading.Lock()

        def run_scraper_thread(platform: str, script: str):
            """Run a scraper in a thread"""
            try:
                self.log(f"Starting {platform} scraper in parallel...")

                # Create wrapper script for this thread
                wrapper_file = Path(f"temp_{platform}_scraper_wrapper.py")

                wrapper_code = textwrap.dedent(f'''
                    import subprocess
                    import sys
                    
                    # Prepare inputs for the scraper (use normalized search terms with spaces)
                    inputs = "{self.search_terms}\\n{num_pages}\\n"
                    
                    # Run the actual scraper with inputs
                    result = subprocess.run(
                        [sys.executable, "{script}"],
                        input=inputs,
                        text=True,
                        capture_output=True,
                        encoding='utf-8'
                    )
                    
                    # Forward the output
                    if result.stdout:
                        print(f"[{platform.upper()}] {{result.stdout}}")
                    if result.stderr:
                        print(f"[{platform.upper()}] {{result.stderr}}", file=sys.stderr)
                    
                    # Exit with same code
                    sys.exit(result.returncode)
                ''')

                with open(wrapper_file, 'w', encoding='utf-8') as f:
                    f.write(wrapper_code)

                # Run the wrapper script
                result = subprocess.run(
                    [sys.executable, str(wrapper_file)],
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )

                # Clean up wrapper
                if wrapper_file.exists():
                    wrapper_file.unlink()

                # Store result
                with results_lock:
                    success = result.returncode == 0
                    results[platform] = success

                    if success:
                        self.log(f"✓ {platform.capitalize()} scraping completed successfully")
                    else:
                        self.log(f"✗ {platform.capitalize()} scraping failed with return code {result.returncode}",
                                 "ERROR")
                        if result.stderr:
                            self.log(f"{platform.capitalize()} error: {result.stderr}", "ERROR")

                    # Log output if verbose
                    if self.verbose and result.stdout:
                        for line in result.stdout.split('\n'):
                            if line.strip():
                                self.log(f"[{platform.upper()}] {line}")

                return success

            except Exception as e:
                self.log(f"✗ Failed to run {platform} scraper: {str(e)}", "ERROR")
                with results_lock:
                    results[platform] = False
                return False

        # Run scrapers in parallel
        self.log("Running scrapers in parallel...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both scraping tasks
            futures = {
                executor.submit(run_scraper_thread, 'temu', 'TemuScraper.py'): 'temu',
                executor.submit(run_scraper_thread, 'amazon', 'AmazonScraper.py'): 'amazon'
            }

            # Wait for both to complete
            for future in as_completed(futures):
                platform = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self.log(f"Exception in {platform} scraper thread: {e}", "ERROR")

        elapsed = time.time() - start_time
        self.log(f"Parallel scraping completed in {elapsed:.1f} seconds")

        return results['temu'], results['amazon']

    def update_data_organizers(self):
        """Create wrapper scripts for data organizers with correct file paths"""
        # Update Temu organizer
        temu_wrapper = Path("temp_temu_organizer.py")
        temu_code = textwrap.dedent(f'''
            import sys
            import json
            from pathlib import Path
            
            # Load config
            with open("pipeline_config.json", 'r') as f:
                config = json.load(f)
            
            # Import and modify the organizer
            sys.path.insert(0, ".")
            import TemuDataOrganizer
            
            # Override the file paths
            TemuDataOrganizer.CSV_FILE = Path(config["temu_csv"])
            TemuDataOrganizer.OUTPUT_FILE = Path(config["temu_json"])
            
            # Run the main function
            TemuDataOrganizer.main()
        ''')

        # Update Amazon organizer
        amazon_wrapper = Path("temp_amazon_organizer.py")
        amazon_code = textwrap.dedent(f'''
            import sys
            import json
            from pathlib import Path
            
            # Load config
            with open("pipeline_config.json", 'r') as f:
                config = json.load(f)
            
            # Import and modify the organizer
            sys.path.insert(0, ".")
            import AmazonDataOrganizer
            
            # Override the file paths
            AmazonDataOrganizer.CSV_FILE = Path(config["amazon_csv"])
            AmazonDataOrganizer.OUTPUT_FILE = Path(config["amazon_json"])
            
            # Run the main function
            AmazonDataOrganizer.main()
        ''')

        try:
            with open(temu_wrapper, 'w', encoding='utf-8') as f:
                f.write(temu_code)
            with open(amazon_wrapper, 'w', encoding='utf-8') as f:
                f.write(amazon_code)

            return str(temu_wrapper), str(amazon_wrapper)
        except Exception as e:
            self.log(f"Error creating wrapper scripts: {e}", "ERROR")
            return None, None

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

                    # Run matching with filtered files AND CORRECT IMAGE DIRECTORIES
                    results = matcher.match_all_products(
                        "{temp_temu}",
                        "{temp_amazon}",
                        temu_img_dir="temu_{self.search_slug}_imgs",
                        amazon_img_dir="amazon_{self.search_slug}_imgs"
                    )

                    # Save results (use temp file)
                    matcher.save_results(results, "temp_matching_results.json")

                    # Generate LLM review batch if needed
                    review_needed = sum(1 for r in results if r.needs_review)
                    if review_needed > 0:
                        # Load full products from filtered data
                        with open("{temp_temu}", 'r') as f:
                            temu_full = json.load(f)['all_products']
                        with open("{temp_amazon}", 'r') as f:
                            amazon_full = json.load(f)['all_products']

                        # Pass correct image directories to generate_llm_review_batch
                        matcher.generate_llm_review_batch(
                            results,
                            temu_full,
                            amazon_full,
                            f"llm_review_batch_{self.search_slug}.json",
                            temu_img_dir="temu_{self.search_slug}_imgs",
                            amazon_img_dir="amazon_{self.search_slug}_imgs"
                        )

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
        self.log(f"Search Terms: '{self.search_terms}'")
        self.log(f"File Slug: '{self.search_slug}'")
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

        # Step 1 & 2: Scraping (run in parallel)
        if not self.skip_scraping:
            self.log("\n--- STEP 1 & 2: Web Scraping (Parallel) ---")

            # Ask for number of pages
            try:
                num_pages = int(input("How many pages to scrape from each site? (default: 1): ") or "1")
            except ValueError:
                num_pages = 1
                self.log("Invalid input, using 1 page")

            # Run scrapers in parallel
            temu_success, amazon_success = self.run_scrapers_parallel(num_pages)

            # Check results
            if not temu_success:
                self.log("Temu scraping failed. Continue anyway? (y/n): ", "WARNING")
                if input().lower() != 'y':
                    return False

            if not amazon_success:
                self.log("Amazon scraping failed. Continue anyway? (y/n): ", "WARNING")
                if input().lower() != 'y':
                    return False

            # Check output files
            self.check_output_files([self.temu_csv, self.amazon_csv], "Scraping")
        else:
            self.log("\n--- Skipping scraping (using existing CSVs) ---")

            # CHECK IF REQUIRED CSV FILES EXIST WHEN SKIPPING SCRAPING
            missing_files = []
            if not self.temu_csv.exists():
                missing_files.append(str(self.temu_csv))
            if not self.amazon_csv.exists():
                missing_files.append(str(self.amazon_csv))

            if missing_files:
                self.log("\n" + "=" * 60, "ERROR")
                self.log("ERROR: Cannot skip scraping - required CSV files are missing:", "ERROR")
                for file in missing_files:
                    self.log(f"  - {file}", "ERROR")

                # Show available search terms
                available_terms = self.find_available_data_files()
                if available_terms:
                    self.log("\nAvailable search terms with existing data:", "ERROR")
                    for term in available_terms:
                        self.log(f"  - {term}", "ERROR")
                    self.log("\nTo use existing data, run with one of these search terms:", "ERROR")
                    self.log(f'  python {sys.argv[0]} --search "{available_terms[0]}" --skip-scraping', "ERROR")
                else:
                    self.log("\nNo existing data files found in the current directory.", "ERROR")

                self.log("\nTo proceed, either:", "ERROR")
                self.log("  1. Run without --skip-scraping to scrape fresh data", "ERROR")
                self.log("  2. Use a search term that has existing data files", "ERROR")
                self.log(f"\nExpected files for search term '{self.search_terms}':", "ERROR")
                self.log(f"  - {self.temu_csv}", "ERROR")
                self.log(f"  - {self.amazon_csv}", "ERROR")
                self.log("=" * 60, "ERROR")
                return False

        # Step 3 & 4: Data Organization
        self.log("\n--- STEP 3 & 4: Data Organization ---")

        # Create wrapper scripts with dynamic paths
        temu_wrapper, amazon_wrapper = self.update_data_organizers()

        if not temu_wrapper or not amazon_wrapper:
            self.log("Failed to create organizer wrappers", "ERROR")
            return False

        if not self.run_script(temu_wrapper, "Organizing Temu data"):
            return False

        if not self.run_script(amazon_wrapper, "Organizing Amazon data"):
            return False

        # Clean up wrapper scripts
        for wrapper in [temu_wrapper, amazon_wrapper]:
            try:
                Path(wrapper).unlink()
            except:
                pass

        if not self.check_output_files([self.temu_json, self.amazon_json], "Data organization"):
            # Additional helpful error message
            self.log("\n" + "=" * 60, "ERROR")
            self.log("ERROR: Data organization failed to create JSON files.", "ERROR")
            self.log("This usually happens when:", "ERROR")
            self.log("  1. The CSV files are empty or corrupted", "ERROR")
            self.log("  2. The data organizer scripts are missing or have errors", "ERROR")
            self.log("  3. There's insufficient disk space", "ERROR")
            self.log("\nPlease check the logs above for specific error messages.", "ERROR")
            self.log("=" * 60, "ERROR")
            return False

        # Step 5: Incremental Product Matching
        self.log("\n--- STEP 5: Incremental Product Matching ---")

        # Check if JSON files exist before trying to load
        if not self.temu_json.exists() or not self.amazon_json.exists():
            self.log("\n" + "=" * 60, "ERROR")
            self.log("ERROR: Required JSON analysis files are missing:", "ERROR")
            if not self.temu_json.exists():
                self.log(f"  - {self.temu_json}", "ERROR")
            if not self.amazon_json.exists():
                self.log(f"  - {self.amazon_json}", "ERROR")

            self.log("\nThese files should have been created by the data organization step.", "ERROR")

            # Check if CSVs exist but JSONs don't
            if self.temu_csv.exists() and self.amazon_csv.exists():
                self.log("\nThe CSV files exist but JSON files are missing.", "ERROR")
                self.log("Try running the pipeline again without --skip-scraping,", "ERROR")
                self.log("or check if the data organizer scripts are working correctly.", "ERROR")

            self.log("=" * 60, "ERROR")
            return False

        # Load organized data
        try:
            with open(self.temu_json, 'r', encoding='utf-8') as f:
                temu_data = json.load(f)
            temu_products = temu_data.get('all_products', [])

            with open(self.amazon_json, 'r', encoding='utf-8') as f:
                amazon_data = json.load(f)
            amazon_products = amazon_data.get('all_products', [])

            # Check if products were loaded
            if not temu_products and not amazon_products:
                self.log("\n" + "=" * 60, "ERROR")
                self.log("ERROR: No products found in the JSON files.", "ERROR")
                self.log("The data files exist but appear to be empty.", "ERROR")
                self.log("This might happen if:", "ERROR")
                self.log("  1. The original scraping returned no results", "ERROR")
                self.log("  2. The search terms didn't match any products", "ERROR")
                self.log("  3. There was an error during data organization", "ERROR")
                self.log("\nTry running with different search terms or without --skip-scraping.", "ERROR")
                self.log("=" * 60, "ERROR")
                return False

        except json.JSONDecodeError as e:
            self.log("\n" + "=" * 60, "ERROR")
            self.log(f"ERROR: Failed to parse JSON files - they may be corrupted.", "ERROR")
            self.log(f"JSON Error: {e}", "ERROR")
            self.log("\nTry deleting the JSON files and running the pipeline again.", "ERROR")
            self.log("=" * 60, "ERROR")
            return False
        except Exception as e:
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

                # Save merged results (use category-specific name for LLM compatibility)
                merged_file = f"matching_results_{self.search_slug}.json"
                try:
                    with open(merged_file, 'w', encoding='utf-8') as f:
                        json.dump(merged_results, f, indent=2)
                except Exception as e:
                    self.log(f"Error saving merged results: {e}", "ERROR")
                    return False

                self.log(f"Incremental matching completed successfully. Results saved to {merged_file}")

            finally:
                # Always clean up temp file, even if something failed
                if temp_results_file.exists():
                    try:
                        temp_results_file.unlink()
                    except Exception as e:
                        self.log(f"Warning: Could not delete temp results file: {e}", "WARNING")

        # Check for matching results - FIX: Use category-specific filename
        category_specific_results = f"matching_results_{self.search_slug}.json"
        if not self.check_output_files([category_specific_results], "Product matching"):
            return False

        # Step 6: LLM Enhancement (optional)
        if not self.skip_llm:
            self.log("\n--- STEP 6: LLM Enhancement ---")

            # Only run LLM on new/updated matches if needed
            try:
                # Load from the category-specific file we just saved
                merged_file = f"matching_results_{self.search_slug}.json"
                with open(merged_file, 'r', encoding='utf-8') as f:
                    current_results = json.load(f)

                new_match_count = current_results.get('metadata', {}).get('new_matches', 0)
                if new_match_count > 0:
                    self.log(f"Running LLM analysis on {new_match_count} new matches")
                    # Pass category as input to avoid prompt hang and use correct file
                    category_input = self.search_slug + "\n"
                    if not self.run_script(
                            "ImageFinalPassLLM.py",
                            "Running LLM re-ranking on matches",
                            input_text=category_input
                    ):
                        self.log("LLM enhancement failed – continuing pipeline", "WARNING")
                else:
                    self.log("No new matches found—skipping LLM enhancement")
            except Exception as e:
                self.log(f"Error in LLM enhancement: {e}", "WARNING")
        else:
            self.log("\n--- Skipping LLM enhancement ---")

        # Step 7: Open Dashboard
        if not self.no_dashboard:
            self.log("\n--- STEP 7: Opening Dashboard ---")
            self.serve_dashboard_with_http()
        else:
            self.log("\n--- STEP 7: Skipping Dashboard (API mode) ---")
            self.log("Dashboard available at server URL")

        # Clean up config file
        try:
            Path("pipeline_config.json").unlink()
        except:
            pass

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
            # Try to load LLM analyzed results first (category-specific)
            results_file = f"matching_results_enriched_{self.search_slug}.json"
            if not Path(results_file).exists():
                results_file = f"matching_results_{self.search_slug}.json"

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


def find_existing_search_terms() -> List[str]:
    """Find all existing search terms by looking at CSV files"""
    csv_files = list(Path(".").glob("temu_*.csv"))
    search_terms = []

    for csv_file in csv_files:
        match = re.match(r"temu_(.+)\.csv", csv_file.name)
        if match:
            # Convert underscores back to spaces for display
            term = match.group(1).replace('_', ' ')
            search_terms.append(term)

    return search_terms


def main():
    parser = argparse.ArgumentParser(description="Run the complete Temu-Amazon matching pipeline")
    parser.add_argument("--search", type=str, default=None,
                        help="Search terms for products (e.g., 'baby toys', 'kitchen gadgets')")
    parser.add_argument("--skip-scraping", action="store_true",
                        help="Skip web scraping (use existing CSV files)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM enhancement step")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    parser.add_argument("--http-server", action="store_true",
                        help="Use HTTP server for dashboard instead of file://")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Skip opening dashboard (for API server mode)")  # Add this line
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

    # Get search terms
    if args.search:
        # Normalize the provided search terms
        search_terms = normalize_search_terms(args.search)
    elif not args.skip_scraping:
        # Ask for search terms if not provided and scraping is enabled
        user_input = input("Enter search terms (default: 'baby toys'): ").strip()
        search_terms = normalize_search_terms(user_input) if user_input else "baby toys"
    else:
        # For skipping scraping, try to detect existing files
        print("Looking for existing CSV files...")
        existing_terms = find_existing_search_terms()

        if existing_terms:
            print("Found existing searches:")
            for i, term in enumerate(existing_terms):
                print(f"  {i + 1}. {term}")

            choice = input("Enter number to use existing search, or type new search terms: ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(existing_terms):
                    search_terms = existing_terms[idx]
                else:
                    search_terms = normalize_search_terms(choice)
            except ValueError:
                search_terms = normalize_search_terms(choice) if choice else "baby toys"
        else:
            user_input = input("Enter search terms (default: 'baby toys'): ").strip()
            search_terms = normalize_search_terms(user_input) if user_input else "baby toys"

    if args.step:
        runner = PipelineRunner(verbose=not args.quiet, search_terms=search_terms, no_dashboard=args.no_dashboard)

        if args.step == "scrape":
            num_pages = int(input("How many pages to scrape? (default: 1): ") or "1")
            # Use parallel scraping
            runner.run_scrapers_parallel(num_pages)
        elif args.step == "organize":
            temu_wrapper, amazon_wrapper = runner.update_data_organizers()
            runner.run_script(temu_wrapper, "Organizing Temu data")
            runner.run_script(amazon_wrapper, "Organizing Amazon data")
            for wrapper in [temu_wrapper, amazon_wrapper]:
                try:
                    Path(wrapper).unlink()
                except:
                    pass
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
            use_http_server=args.http_server,
            search_terms=search_terms,
            no_dashboard=args.no_dashboard  # Add this line
        )
        runner.run_pipeline()


if __name__ == "__main__":
    main()
