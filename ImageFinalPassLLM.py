"""
LLM Re-ranking for Product Matches
Re-ranks top product matches using a vision-language model via LM Studio
"""
import json
import base64
import requests
import fire
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LM Studio configuration
LM_STUDIO_URL = "http://localhost:1234"  # Update with your LM Studio URL
DEFAULT_MODEL = None  # Will use whatever model is loaded in LM Studio

@dataclass
class LLMVerdict:
    """Result from LLM analysis of a product pair"""
    same_product: bool
    rationale: str
    image_similarity: float
    processing_time: float

class ProductMatchReranker:
    def __init__(self,
                 lm_studio_url: str = LM_STUDIO_URL,
                 model_name: Optional[str] = DEFAULT_MODEL,
                 image_dirs: Dict[str, Path] = None):
        """
        Initialize the reranker

        Args:
            lm_studio_url: URL of LM Studio server
            model_name: Specific model to use (optional)
            image_dirs: Dict with 'temu' and 'amazon' paths to image directories
        """
        self.lm_studio_url = lm_studio_url
        self.model_name = model_name
        self.image_dirs = image_dirs or {
            'temu': Path('temu_baby_toys_imgs'),
            'amazon': Path('amazon_baby_toys_imgs')
        }

        # Try local SDK first
        self.use_local_sdk = self._check_local_sdk()

    def _check_local_sdk(self) -> bool:
        """Check if local LM Studio SDK is available"""
        # Temporarily disable local SDK due to bosToken issues
        # Uncomment the lines below to re-enable local SDK
        logger.info("Using remote LM Studio API (local SDK disabled)")
        return False

        # try:
        #     import lmstudio as lms
        #     logger.info("Local LM Studio SDK detected")
        #     return True
        # except ImportError:
        #     logger.info("Using remote LM Studio API")
        #     return False

    def select_top_candidates(self,
                            matches: List[Dict],
                            percentage: float = 0.35,
                            min_confidence: float = 0.6) -> List[Dict]:
        """
        Select top percentage of matches for LLM re-ranking

        Args:
            matches: List of match dictionaries
            percentage: Top percentage to select (0-1)
            min_confidence: Minimum confidence threshold

        Returns:
            Selected matches for LLM processing
        """
        # Sort by confidence (descending)
        sorted_matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)

        # Calculate how many to take
        top_n = math.ceil(len(sorted_matches) * percentage)

        # Apply confidence floor
        candidates = []
        for match in sorted_matches[:top_n]:
            if match['confidence'] >= min_confidence:
                candidates.append(match)
            else:
                break  # Since sorted, no point checking further

        logger.info(f"Selected {len(candidates)} candidates from {len(matches)} matches "
                   f"(top {percentage*100}% with confidence >= {min_confidence})")

        return candidates

    def _load_image_as_base64(self, image_path: Path) -> Optional[str]:
        """Load image and convert to base64"""
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return None

        try:
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def _prepare_llm_prompt(self, match: Dict, temu_img_b64: str, amazon_img_b64: str) -> Dict:
        """Prepare the prompt for LLM analysis"""
        # Extract price difference info
        temu_price = match.get('temu_price', 0)
        amazon_price = match.get('amazon_price', 0)
        price_diff = abs(amazon_price - temu_price)
        price_diff_pct = (price_diff / min(temu_price, amazon_price) * 100) if min(temu_price, amazon_price) > 0 else 0

        # Build the prompt
        prompt_text = f"""You are a product-matching assistant. Analyze these two products and determine if they represent the SAME physical product (not just similar category).

Product 1 (Temu):
- Title: {match.get('temu_title', 'Unknown')}
- Price: ${temu_price:.2f}

Product 2 (Amazon):
- Title: {match.get('amazon_title', 'Unknown')}
- Price: ${amazon_price:.2f}

Price difference: ${price_diff:.2f} ({price_diff_pct:.1f}%)

Consider:
1. Visual appearance (shape, color, design, components)
2. Product features and accessories
3. Brand/manufacturer (if visible)
4. Whether price difference is reasonable for same product

Respond with JSON only:
{{
  "same_product": true/false,
  "image_similarity": 0.0-1.0,
  "rationale": "Brief explanation"
}}"""

        # Build message content
        content = [{"type": "text", "text": prompt_text}]

        if temu_img_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{temu_img_b64}"}
            })

        if amazon_img_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{amazon_img_b64}"}
            })

        return {
            "messages": [{
                "role": "user",
                "content": content
            }],
            "max_tokens": 200,
            "temperature": 0.1,  # Low temperature for consistency
            "stream": False
        }

    def _analyze_match_local(self, match: Dict, temu_img_path: Path, amazon_img_path: Path) -> Optional[LLMVerdict]:
        """Analyze using local LM Studio SDK"""
        try:
            import lmstudio as lms

            model = lms.llm(self.model_name) if self.model_name else lms.llm()

            # Prepare images
            images = []
            if temu_img_path.exists():
                images.append(lms.prepare_image(str(temu_img_path)))
            if amazon_img_path.exists():
                images.append(lms.prepare_image(str(amazon_img_path)))

            # Create prompt
            temu_price = match.get('temu_price', 0)
            amazon_price = match.get('amazon_price', 0)
            price_diff = abs(amazon_price - temu_price)
            price_diff_pct = (price_diff / min(temu_price, amazon_price) * 100) if min(temu_price, amazon_price) > 0 else 0

            prompt = f"""You are a product-matching assistant. Analyze these two products and determine if they represent the SAME physical product (not just similar category).

Product 1 (Temu):
- Title: {match.get('temu_title', 'Unknown')}
- Price: ${temu_price:.2f}

Product 2 (Amazon):
- Title: {match.get('amazon_title', 'Unknown')}
- Price: ${amazon_price:.2f}

Price difference: ${price_diff:.2f} ({price_diff_pct:.1f}%)

Consider:
1. Visual appearance (shape, color, design, components)
2. Product features and accessories
3. Brand/manufacturer (if visible)
4. Whether price difference is reasonable for same product

Respond with JSON only:
{{
  "same_product": true/false,
  "image_similarity": 0.0-1.0,
  "rationale": "Brief explanation"
}}"""

            chat = lms.Chat()
            chat.add_user_message(prompt, images=images)

            start_time = time.time()
            response = model.respond(chat)
            processing_time = time.time() - start_time

            # The response is a PredictionResult object, get the text content
            response_text = str(response)

            # Parse JSON from response text
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Try to parse the whole response as JSON
                result = json.loads(response_text)

            return LLMVerdict(
                same_product=result['same_product'],
                rationale=result.get('rationale', ''),
                image_similarity=result.get('image_similarity', 0.0),
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Local SDK error: {e}")
            return None

    def _analyze_match_remote(self, match: Dict, temu_img_b64: str, amazon_img_b64: str) -> Optional[LLMVerdict]:
        """Analyze using remote LM Studio API"""
        try:
            payload = self._prepare_llm_prompt(match, temu_img_b64, amazon_img_b64)
            if self.model_name:
                payload["model"] = self.model_name

            start_time = time.time()
            response = requests.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            processing_time = time.time() - start_time

            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None

            result_text = response.json()['choices'][0]['message']['content']

            # Parse JSON from response
            # Try to extract JSON if wrapped in text
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(result_text)

            return LLMVerdict(
                same_product=result['same_product'],
                rationale=result.get('rationale', ''),
                image_similarity=result.get('image_similarity', 0.0),
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Remote API error for match {match['temu_id']}-{match['amazon_id']}: {e}")
            return None

    def analyze_single_match(self, match: Dict) -> Tuple[Dict, Optional[LLMVerdict]]:
        """Analyze a single match with LLM"""
        temu_id = match['temu_id']
        amazon_id = match['amazon_id']

        # Get image paths
        temu_img_path = self.image_dirs['temu'] / f"{temu_id}.jpg"
        amazon_img_path = self.image_dirs['amazon'] / f"{amazon_id}.jpg"

        # Try local SDK first
        if self.use_local_sdk:
            verdict = self._analyze_match_local(match, temu_img_path, amazon_img_path)
            if verdict:
                return match, verdict

        # Fallback to remote API
        temu_img_b64 = self._load_image_as_base64(temu_img_path)
        amazon_img_b64 = self._load_image_as_base64(amazon_img_path)

        if not temu_img_b64 and not amazon_img_b64:
            logger.warning(f"No images found for match {temu_id}-{amazon_id}, skipping")
            return match, None

        verdict = self._analyze_match_remote(match, temu_img_b64, amazon_img_b64)
        return match, verdict

    def run_llm_batch(self,
                     candidates: List[Dict],
                     batch_size: int = 5,
                     max_workers: int = 3,
                     checkpoint_callback=None) -> List[Tuple[Dict, Optional[LLMVerdict]]]:
        """
        Process candidates in batches with LLM

        Args:
            candidates: List of match candidates
            batch_size: Number to process in parallel
            max_workers: Max concurrent threads
            checkpoint_callback: Function to call after each result (for saving progress)

        Returns:
            List of (match, verdict) tuples
        """
        results = []
        total = len(candidates)
        processed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batches
            future_to_match = {}

            for i in range(0, total, batch_size):
                batch = candidates[i:i+batch_size]
                for match in batch:
                    future = executor.submit(self.analyze_single_match, match)
                    future_to_match[future] = match

                # Process completed futures
                for future in as_completed(future_to_match):
                    match, verdict = future.result()
                    results.append((match, verdict))
                    processed_count += 1

                    if verdict:
                        logger.info(f"Processed {match['temu_id']}-{match['amazon_id']}: "
                                   f"same_product={verdict.same_product} "
                                   f"[{processed_count}/{total}]")
                    else:
                        logger.warning(f"Failed to process {match['temu_id']}-{match['amazon_id']} "
                                     f"[{processed_count}/{total}]")

                    # Call checkpoint callback if provided
                    if checkpoint_callback:
                        checkpoint_callback(results)

                # Rate limiting between batches
                if i + batch_size < total:
                    time.sleep(1)

        return results

    def merge_llm_results(self,
                         original_matches: List[Dict],
                         llm_results: List[Tuple[Dict, Optional[LLMVerdict]]]) -> List[Dict]:
        """Merge LLM verdicts back into matches"""
        # Create lookup for LLM results
        llm_lookup = {}
        for match, verdict in llm_results:
            key = f"{match['temu_id']}_{match['amazon_id']}"
            llm_lookup[key] = verdict

        # Enhance matches with LLM data
        enhanced_matches = []
        for match in original_matches:
            key = f"{match['temu_id']}_{match['amazon_id']}"
            enhanced_match = match.copy()

            if key in llm_lookup and llm_lookup[key]:
                verdict = llm_lookup[key]
                enhanced_match['llm_same_product'] = verdict.same_product
                enhanced_match['llm_rationale'] = verdict.rationale
                enhanced_match['llm_image_similarity'] = verdict.image_similarity
                enhanced_match['llm_processing_time'] = verdict.processing_time
                enhanced_match['llm_processed'] = True
            else:
                enhanced_match['llm_processed'] = False

            enhanced_matches.append(enhanced_match)

        return enhanced_matches

def main(input_file: str = "matching_results.json",
         output_file: str = "matching_results_enriched.json",
         percentage: float = 0.35,
         min_confidence: float = 0.6,
         batch_size: int = 5,
         lm_studio_url: str = LM_STUDIO_URL,
         model_name: Optional[str] = None,
         checkpoint_interval: int = 5):
    """
    Main function to run LLM re-ranking on product matches

    Args:
        input_file: Input JSON file with matches
        output_file: Output file for enriched results
        percentage: Top percentage of matches to process (0-1)
        min_confidence: Minimum confidence threshold
        batch_size: Number of matches to process in parallel
        lm_studio_url: URL of LM Studio server
        model_name: Specific model to use (optional)
        checkpoint_interval: Save progress every N results
    """
    logger.info("=== Product Match LLM Re-ranking ===")

    # Load matches
    logger.info(f"Loading matches from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    matches = data.get('matches', [])
    metadata = data.get('metadata', {})

    # Check for existing checkpoint
    checkpoint_file = Path(output_file).with_suffix('.checkpoint.json')
    existing_results = []
    processed_keys = set()

    if checkpoint_file.exists():
        logger.info(f"Found checkpoint file: {checkpoint_file}")
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                checkpoint_results = checkpoint_data.get('results', [])

                # Convert dicts back to LLMVerdict objects
                for match, verdict_dict in checkpoint_results:
                    if verdict_dict:
                        verdict = LLMVerdict(**verdict_dict)
                    else:
                        verdict = None
                    existing_results.append((match, verdict))
                    key = f"{match['temu_id']}_{match['amazon_id']}"
                    processed_keys.add(key)

                logger.info(f"Loaded {len(existing_results)} previously processed results")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")

    # Initialize reranker
    reranker = ProductMatchReranker(
        lm_studio_url=lm_studio_url,
        model_name=model_name
    )

    # Select candidates
    all_candidates = reranker.select_top_candidates(
        matches,
        percentage=percentage,
        min_confidence=min_confidence
    )

    # Filter out already processed candidates
    candidates = []
    for candidate in all_candidates:
        key = f"{candidate['temu_id']}_{candidate['amazon_id']}"
        if key not in processed_keys:
            candidates.append(candidate)

    logger.info(f"Found {len(candidates)} new candidates to process (skipping {len(all_candidates) - len(candidates)} already processed)")

    if not candidates:
        logger.info("No new candidates to process")
    else:
        # Define checkpoint callback
        all_results = existing_results.copy()
        results_since_checkpoint = 0

        def save_checkpoint(current_results):
            nonlocal all_results, results_since_checkpoint
            results_since_checkpoint += 1

            if results_since_checkpoint >= checkpoint_interval:
                # Combine existing and new results
                all_results = existing_results + current_results

                # Convert LLMVerdict objects to dicts for JSON serialization
                serializable_results = []
                for match, verdict in all_results:
                    if verdict:
                        # Convert dataclass to dict
                        verdict_dict = asdict(verdict)
                    else:
                        verdict_dict = None
                    serializable_results.append((match, verdict_dict))

                # Save checkpoint
                checkpoint_data = {
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'total_processed': len(all_results),
                    'results': serializable_results
                }

                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2)

                logger.info(f"Checkpoint saved: {len(all_results)} total results")
                results_since_checkpoint = 0

        # Process with LLM
        logger.info(f"Processing {len(candidates)} candidates with LLM...")
        new_results = reranker.run_llm_batch(
            candidates,
            batch_size=batch_size,
            checkpoint_callback=save_checkpoint
        )

        # Combine all results
        all_results = existing_results + new_results

    # Merge results
    enhanced_matches = reranker.merge_llm_results(matches, all_results)

    # Calculate statistics
    llm_stats = {
        'total_processed': sum(1 for m in enhanced_matches if m.get('llm_processed', False)),
        'llm_same': sum(1 for m in enhanced_matches if m.get('llm_same_product', False)),
        'llm_different': sum(1 for m in enhanced_matches if m.get('llm_processed', False) and not m.get('llm_same_product', True)),
        'average_processing_time': sum(m.get('llm_processing_time', 0) for m in enhanced_matches if m.get('llm_processed', False)) / max(1, sum(1 for m in enhanced_matches if m.get('llm_processed', False)))
    }

    # Update metadata
    metadata.update({
        'llm_enrichment': {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'candidates_selected': len(all_candidates),
            'percentage_used': percentage,
            'min_confidence_used': min_confidence,
            **llm_stats
        }
    })

    # Save final results
    output_data = {
        'metadata': metadata,
        'matches': enhanced_matches
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    # Remove checkpoint file on successful completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info("Removed checkpoint file after successful completion")

    logger.info(f"\nEnrichment complete! Results saved to {output_file}")
    logger.info(f"Processed: {llm_stats['total_processed']}")
    logger.info(f"Same product: {llm_stats['llm_same']}")
    logger.info(f"Different product: {llm_stats['llm_different']}")
    logger.info(f"Average processing time: {llm_stats['average_processing_time']:.2f}s per match")

if __name__ == "__main__":
    fire.Fire(main)
