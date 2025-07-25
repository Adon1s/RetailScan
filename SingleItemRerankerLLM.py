"""
Single Item LLM Re-ranker
Re-ranks product matches against a user's input item using vision-language model
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
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LM Studio configuration
LM_STUDIO_URL = "http://localhost:1234"
DEFAULT_MODEL = None  # Will use whatever model is loaded in LM Studio


@dataclass
class SingleItemLLMVerdict:
    """Result from LLM analysis comparing product to user's item"""
    same_product: bool
    similarity_score: float  # 0-1 scale
    rationale: str
    key_differences: List[str]
    processing_time: float


def sanitize_filename(filename: str) -> str:
    """Sanitize string for use in filenames"""
    sanitized = re.sub(r'[^\w\s-]', '', filename)
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    return sanitized.lower()


class SingleItemLLMReranker:
    def __init__(self,
                 lm_studio_url: str = LM_STUDIO_URL,
                 model_name: Optional[str] = DEFAULT_MODEL,
                 category_name: str = None):
        """
        Initialize the single item reranker

        Args:
            lm_studio_url: URL of LM Studio server
            model_name: Specific model to use (optional)
            category_name: Product category for finding image directories
        """
        self.lm_studio_url = lm_studio_url
        self.model_name = model_name
        self.category_slug = sanitize_filename(category_name) if category_name else 'default'

        # Image directories based on category
        self.image_dirs = {
            'temu': Path(f'temu_{self.category_slug}_imgs'),
            'amazon': Path(f'amazon_{self.category_slug}_imgs')
        }

        # User item cache
        self.user_item_image_b64 = None
        self.user_item_data = None

        logger.info(f"Initialized Single Item LLM Reranker for category: {self.category_slug}")

    def load_user_item(self, user_item: Dict) -> bool:
        """Load and cache the user's input item"""
        self.user_item_data = user_item

        # Load user image if available
        image_path = user_item.get('image_path', '')
        if image_path:
            self.user_item_image_b64 = self._load_image_as_base64_from_path(image_path)
            if self.user_item_image_b64:
                logger.info("User item image loaded successfully")
                return True
            else:
                logger.warning("Failed to load user item image")
                return False
        else:
            logger.warning("No image path provided for user item")
            return False

    def _load_image_as_base64_from_path(self, image_path_or_url: str) -> Optional[str]:
        """Load image from file path or URL and convert to base64"""
        try:
            # Check if it's a URL
            if image_path_or_url.startswith('http'):
                response = requests.get(image_path_or_url, timeout=10)
                response.raise_for_status()
                return base64.b64encode(response.content).decode('utf-8')
            else:
                # Local file
                with open(image_path_or_url, 'rb') as img_file:
                    return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error loading image {image_path_or_url}: {e}")
            return None

    def _load_image_as_base64(self, image_path: Path) -> Optional[str]:
        """Load image from Path and convert to base64"""
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return None

        try:
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def _create_comparison_prompt(self, match: Dict) -> str:
        """Create prompt for comparing match against user item"""
        user_title = self.user_item_data.get('title', 'Unknown')
        match_title = match.get('title', 'Unknown')
        match_price = match.get('price', 0)
        match_platform = match.get('platform', 'Unknown')

        # Get initial confidence scores
        confidence = match.get('confidence', 0)
        fuzzy_score = match.get('scores', {}).get('fuzzy', 0)
        embedding_score = match.get('scores', {}).get('embedding', 0)
        image_score = match.get('scores', {}).get('image')

        # Format image score properly
        image_score_str = f"{image_score:.3f}" if image_score is not None else "N/A"

        return f"""You are a product comparison expert. Compare these two products and determine if they are the SAME product or just similar.

USER'S PRODUCT:
- Title: {user_title}
- [User provided image shown]

CANDIDATE PRODUCT (from {match_platform.upper()}):
- Title: {match_title}
- Price: ${match_price:.2f}
- Platform: {match_platform}
- Initial Match Scores:
  * Text similarity: {fuzzy_score}% fuzzy, {embedding_score:.3f} semantic
  * Image similarity: {image_score_str}
  * Overall confidence: {confidence:.1%}

Analyze both products carefully considering:
1. Visual appearance - shape, color, design, components
2. Product type and category
3. Features and specifications mentioned
4. Brand indicators (if visible)
5. Whether differences could be due to:
   - Different product variants/colors
   - Packaging differences
   - Photo angle/lighting
   - Minor design updates

Respond with JSON only:
{{
  "same_product": true/false,
  "similarity_score": 0.0-1.0,
  "rationale": "Brief explanation of your decision",
  "key_differences": ["difference 1", "difference 2", ...] or [] if same
}}

Be strict: only mark as same_product=true if you're confident they are the exact same product."""

    def _prepare_llm_request(self, match: Dict, match_img_b64: Optional[str]) -> Dict:
        """Prepare the LLM API request"""
        prompt_text = self._create_comparison_prompt(match)

        # Build message content
        content = [{"type": "text", "text": prompt_text}]

        # Add user item image first
        if self.user_item_image_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self.user_item_image_b64}"}
            })

        # Add match product image
        if match_img_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{match_img_b64}"}
            })

        return {
            "messages": [{
                "role": "user",
                "content": content
            }],
            "max_tokens": 300,
            "temperature": 0.1,
            "stream": False
        }

    def analyze_single_match(self, match: Dict) -> Tuple[Dict, Optional[SingleItemLLMVerdict]]:
        """Analyze a single match against the user's item"""
        platform = match.get('platform', '')
        product_id = match.get('product_id', '')

        # Get product image
        img_path = self.image_dirs.get(platform) / f"{product_id}.jpg"
        match_img_b64 = self._load_image_as_base64(img_path) if img_path else None

        if not match_img_b64 and not self.user_item_image_b64:
            logger.warning(f"No images available for comparison: {product_id}")
            return match, None

        try:
            # Prepare and send request
            payload = self._prepare_llm_request(match, match_img_b64)
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
                return match, None

            result_text = response.json()['choices'][0]['message']['content']

            # Parse JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(result_text)

            verdict = SingleItemLLMVerdict(
                same_product=result.get('same_product', False),
                similarity_score=float(result.get('similarity_score', 0.0)),
                rationale=result.get('rationale', ''),
                key_differences=result.get('key_differences', []),
                processing_time=processing_time
            )

            return match, verdict

        except Exception as e:
            logger.error(f"Error analyzing match {product_id}: {e}")
            return match, None

    def rerank_matches(self,
                       matches: List[Dict],
                       max_to_process: int = 20,
                       batch_size: int = 5,
                       max_workers: int = 3) -> List[Dict]:
        """
        Re-rank matches using LLM comparison against user item

        Args:
            matches: List of matches to re-rank
            max_to_process: Maximum number of matches to process with LLM
            batch_size: Number of matches to process in parallel
            max_workers: Number of parallel workers

        Returns:
            Re-ranked list of matches with LLM verdicts
        """
        # Take top matches to process
        candidates = matches[:max_to_process]
        logger.info(f"Processing {len(candidates)} candidates with LLM")

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process in batches
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i + batch_size]
                future_to_match = {}

                for match in batch:
                    future = executor.submit(self.analyze_single_match, match)
                    future_to_match[future] = match

                for future in as_completed(future_to_match):
                    match, verdict = future.result()

                    # Enhance match with LLM verdict
                    enhanced_match = match.copy()
                    if verdict:
                        enhanced_match['llm_same_product'] = verdict.same_product
                        enhanced_match['llm_similarity_score'] = verdict.similarity_score
                        enhanced_match['llm_rationale'] = verdict.rationale
                        enhanced_match['llm_key_differences'] = verdict.key_differences
                        enhanced_match['llm_processing_time'] = verdict.processing_time
                        enhanced_match['llm_processed'] = True

                        # Update confidence based on LLM verdict
                        if verdict.same_product:
                            # Boost confidence for same products
                            enhanced_match['final_confidence'] = min(
                                enhanced_match['confidence'] * 1.2,
                                verdict.similarity_score
                            )
                        else:
                            # Reduce confidence for different products
                            enhanced_match['final_confidence'] = enhanced_match['confidence'] * verdict.similarity_score

                        logger.info(f"Processed {match['product_id']}: "
                                    f"same={verdict.same_product}, "
                                    f"similarity={verdict.similarity_score:.2f}")
                    else:
                        enhanced_match['llm_processed'] = False
                        enhanced_match['final_confidence'] = enhanced_match['confidence']

                    results.append(enhanced_match)

                # Brief pause between batches
                if i + batch_size < len(candidates):
                    time.sleep(0.5)

        # Add unprocessed matches
        processed_ids = {m['product_id'] for m in results}
        for match in matches[max_to_process:]:
            if match['product_id'] not in processed_ids:
                match['llm_processed'] = False
                match['final_confidence'] = match['confidence']
                results.append(match)

        # Re-sort by final confidence
        results.sort(key=lambda x: x.get('final_confidence', 0), reverse=True)

        return results

    def process_single_item_matches(self,
                                    input_file: str,
                                    output_file: str = None,
                                    max_to_process: int = 20,
                                    batch_size: int = 5) -> Dict:
        """
        Main method to process single item matches with LLM re-ranking

        Args:
            input_file: JSON file from single_item_matcher.py
            output_file: Output file for enriched results
            max_to_process: Number of top matches to process with LLM
            batch_size: Batch size for parallel processing

        Returns:
            Enhanced results dictionary
        """
        # Load input matches
        logger.info(f"Loading matches from {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract components
        metadata = data.get('metadata', {})
        user_item = data.get('input_item', {})
        matches = data.get('matches', [])

        logger.info(f"Found {len(matches)} matches to process")

        # Load user item
        if not self.load_user_item(user_item):
            logger.warning("Proceeding without user item image")

        # Re-rank with LLM
        enhanced_matches = self.rerank_matches(
            matches,
            max_to_process=max_to_process,
            batch_size=batch_size
        )

        # Calculate statistics
        llm_stats = {
            'total_processed': sum(1 for m in enhanced_matches if m.get('llm_processed', False)),
            'llm_same_product': sum(1 for m in enhanced_matches if m.get('llm_same_product', False)),
            'llm_different_product': sum(
                1 for m in enhanced_matches
                if m.get('llm_processed', False) and not m.get('llm_same_product', True)
            ),
            'average_similarity': sum(
                m.get('llm_similarity_score', 0) for m in enhanced_matches
                if m.get('llm_processed', False)
            ) / max(1, sum(1 for m in enhanced_matches if m.get('llm_processed', False))),
            'average_processing_time': sum(
                m.get('llm_processing_time', 0) for m in enhanced_matches
                if m.get('llm_processed', False)
            ) / max(1, sum(1 for m in enhanced_matches if m.get('llm_processed', False)))
        }

        # Update metadata
        metadata['llm_reranking'] = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'max_processed': max_to_process,
            **llm_stats
        }

        # Prepare final results
        results = {
            'metadata': metadata,
            'input_item': user_item,
            'matches': enhanced_matches
        }

        # Save results
        if not output_file:
            base_name = Path(input_file).stem
            output_file = f"{base_name}_llm_reranked.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to {output_file}")
        logger.info(f"LLM Processing Summary:")
        logger.info(f"  - Total processed: {llm_stats['total_processed']}")
        logger.info(f"  - Same product: {llm_stats['llm_same_product']}")
        logger.info(f"  - Different product: {llm_stats['llm_different_product']}")
        logger.info(f"  - Average similarity: {llm_stats['average_similarity']:.2f}")
        logger.info(f"  - Average processing time: {llm_stats['average_processing_time']:.2f}s")

        return results


def main(input_file: str = None,
         output_file: str = None,
         category: str = None,
         max_to_process: int = 20,
         batch_size: int = 5,
         lm_studio_url: str = LM_STUDIO_URL,
         model_name: Optional[str] = None):
    """
    Main function to run LLM re-ranking on single item matches

    Args:
        input_file: Input JSON file from single_item_matcher
        output_file: Output file for enriched results
        category: Product category (for finding images)
        max_to_process: Number of top matches to process with LLM
        batch_size: Number of matches to process in parallel
        lm_studio_url: URL of LM Studio server
        model_name: Specific model to use (optional)
    """
    logger.info("=== Single Item LLM Re-ranker ===")

    # Interactive mode if no input file specified
    if not input_file:
        print("\nEnter the single item matches file to process:")
        print("(or press Enter to use the most recent)")

        # Find most recent single_item_matches file
        matches_files = sorted(Path('.').glob('single_item_matches_*.json'),
                               key=lambda x: x.stat().st_mtime,
                               reverse=True)

        if matches_files:
            print(f"Most recent: {matches_files[0]}")
            user_input = input("File path: ").strip()
            input_file = user_input if user_input else str(matches_files[0])
        else:
            input_file = input("File path: ").strip()

    # Check if input file exists
    if not Path(input_file).exists():
        logger.error(f"❌ Error: {input_file} not found!")
        return

    # Extract category from filename if not provided
    if not category:
        # Try to extract from filename pattern
        match = re.search(r'single_item_matches_([^_]+)_', input_file)
        if match:
            category = match.group(1)
            logger.info(f"Detected category: {category}")
        else:
            print("\nEnter the product category (for finding product images):")
            category = input("Category: ").strip()
            category = sanitize_filename(category)

    # Initialize reranker
    reranker = SingleItemLLMReranker(
        lm_studio_url=lm_studio_url,
        model_name=model_name,
        category_name=category
    )

    # Process matches
    results = reranker.process_single_item_matches(
        input_file=input_file,
        output_file=output_file,
        max_to_process=max_to_process,
        batch_size=batch_size
    )

    # Display top results
    print("\n" + "=" * 80)
    print("TOP 10 MATCHES AFTER LLM RE-RANKING:")
    print("=" * 80)

    for i, match in enumerate(results['matches'][:10], 1):
        print(f"\n{i}. [{match['platform'].upper()}] {match['title']}")
        print(f"   ID: {match['product_id']} | Price: ${match['price']:.2f}")
        print(f"   Initial confidence: {match['confidence']:.1%}")

        if match.get('llm_processed'):
            same = "✓ SAME PRODUCT" if match['llm_same_product'] else "✗ DIFFERENT"
            print(f"   LLM verdict: {same} (similarity: {match['llm_similarity_score']:.1%})")
            print(f"   Final confidence: {match.get('final_confidence', 0):.1%}")
            print(f"   Rationale: {match['llm_rationale'][:100]}...")

            if match.get('llm_key_differences'):
                print(f"   Differences: {', '.join(match['llm_key_differences'][:2])}")
        else:
            print(f"   LLM: Not processed")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    fire.Fire(main)