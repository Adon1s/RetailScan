"""
Temu-Amazon Product Matcher
Matches products between Temu and Amazon using multiple similarity methods
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime

# Core matching libraries
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
import requests
from io import BytesIO
import sys

# Force UTF-8 output so Windows console won’t choke on ✓, ✗, etc.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# Optional: CLIP for image similarity
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    clip = None  # Define clip as None to avoid NameError
    CLIP_AVAILABLE = False
    print("CLIP not installed. Image matching will be disabled.")
    print("Install with: pip install git+https://github.com/openai/CLIP.git")


@dataclass
class MatchResult:
    """Represents a product match between Temu and Amazon"""
    temu_id: str
    amazon_id: str
    confidence: float
    match_method: str
    fuzzy_score: float
    embedding_score: float = 0.0
    image_score: float = 0.0
    verdict_reason: str = ""
    needs_review: bool = False
    # Add these new fields for dashboard compatibility
    temu_title: str = ""
    amazon_title: str = ""
    temu_price: float = 0.0
    amazon_price: float = 0.0


class ProductMatcher:
    def __init__(self, use_images: bool = True):
        """Initialize the matcher with models"""
        print("Initializing ProductMatcher...")

        # Sentence transformer for semantic matching
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # CLIP for image matching (if available and requested)
        print(f"  › use_images flag: {use_images}, CLIP_AVAILABLE: {CLIP_AVAILABLE}")
        self.use_images = use_images and CLIP_AVAILABLE
        if self.use_images:
            assert clip is not None  # Tell IDE that clip cannot be None here
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
            print("CLIP model loaded for image matching")

        # Cache for embeddings
        self.title_embeddings_cache = {}
        self.image_embeddings_cache = {}

        # Updated thresholds for better matching
        self.FUZZY_IMMEDIATE_ACCEPT = 90  # Lowered from 93
        self.FUZZY_CANDIDATE_MIN = 40     # Lowered significantly from 85
        self.EMBEDDING_MIN = 0.70          # Lowered from 0.80
        self.IMAGE_MIN = 0.85              # Lowered from 0.88

    def clean_title(self, title: str) -> str:
        """Clean and normalize product title"""
        # Remove special characters but keep spaces and numbers
        title = re.sub(r'[^\w\s-]', ' ', title.lower())
        # Remove extra spaces
        title = ' '.join(title.split())
        # Remove common filler words only if title is long enough
        filler_words = ['for', 'with', 'and', 'the', 'in', 'on', 'at', 'to', 'a', 'an']
        words = title.split()
        if len(words) > 5:  # Only remove filler words from longer titles
            cleaned = [w for w in words if w not in filler_words]
            return ' '.join(cleaned)
        return title

    def get_title_embedding(self, title: str) -> np.ndarray:
        """Get sentence embedding for title (cached)"""
        if title not in self.title_embeddings_cache:
            self.title_embeddings_cache[title] = self.sentence_model.encode(title)
        return self.title_embeddings_cache[title]

    def get_image_embedding(self, image_path_or_url: str) -> Optional[np.ndarray]:
        """Get CLIP embedding for image (cached)"""
        if not self.use_images:
            return None

        if image_path_or_url in self.image_embeddings_cache:
            return self.image_embeddings_cache[image_path_or_url]

        try:
            # Load image
            if image_path_or_url.startswith('http'):
                response = requests.get(image_path_or_url, timeout=10)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path_or_url)

            # Process with CLIP
            image_input = self.clip_preprocess(image).unsqueeze(0)
            if torch.cuda.is_available():
                image_input = image_input.cuda()

            with torch.no_grad():
                embedding = self.clip_model.encode_image(image_input)
                embedding = embedding.cpu().numpy().flatten()
                embedding = embedding / np.linalg.norm(embedding)  # Normalize

            self.image_embeddings_cache[image_path_or_url] = embedding
            return embedding

        except Exception as e:
            print(f"Error processing image {image_path_or_url}: {e}")
            return None

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def match_single_product(self, temu_product: Dict, amazon_products: List[Dict],
                           temu_img_dir: Path, amazon_img_dir: Path) -> MatchResult:
        """Match a single Temu product against all Amazon products"""
        temu_id = temu_product['temu_id']
        temu_title = temu_product['title']
        temu_title_clean = self.clean_title(temu_title)

        # Prepare Amazon titles
        amazon_titles = [p['title'] for p in amazon_products]
        amazon_titles_clean = [self.clean_title(t) for t in amazon_titles]

        # Step 1: Fuzzy matching - try multiple scorers
        fuzzy_matches_token = process.extract(
            temu_title_clean,
            amazon_titles_clean,
            scorer=fuzz.token_set_ratio,
            limit=10  # Get more candidates
        )

        # Also try partial ratio for substring matches
        fuzzy_matches_partial = process.extract(
            temu_title_clean,
            amazon_titles_clean,
            scorer=fuzz.partial_ratio,
            limit=10
        )

        # Combine and deduplicate matches
        all_matches = {}
        for match_str, score, idx in fuzzy_matches_token:
            all_matches[idx] = max(all_matches.get(idx, 0), score)
        for match_str, score, idx in fuzzy_matches_partial:
            all_matches[idx] = max(all_matches.get(idx, 0), score)

        # Sort by score
        sorted_matches = sorted(all_matches.items(), key=lambda x: x[1], reverse=True)[:10]

        if not sorted_matches:
            # Fallback if no matches at all
            return MatchResult(
                temu_id=temu_id,
                amazon_id=amazon_products[0]['amazon_id'],
                confidence=0.0,
                match_method="no_match",
                fuzzy_score=0.0,
                verdict_reason="No fuzzy matches found",
                needs_review=True,
                temu_title=temu_product['title'],
                amazon_title=amazon_products[0]['title'],
                temu_price=float(temu_product.get('price', 0)),
                amazon_price=float(amazon_products[0].get('price', 0))
            )

        # Get best match info
        best_fuzzy_idx, best_fuzzy_score = sorted_matches[0]

        # Check for immediate acceptance
        if best_fuzzy_score >= self.FUZZY_IMMEDIATE_ACCEPT:
            return MatchResult(
                temu_id=temu_id,
                amazon_id=amazon_products[best_fuzzy_idx]['amazon_id'],
                confidence=best_fuzzy_score / 100,
                match_method="fuzzy_immediate",
                fuzzy_score=best_fuzzy_score,
                verdict_reason=f"Very high fuzzy match score ({best_fuzzy_score}%)",
                temu_title=temu_product['title'],
                amazon_title=amazon_products[best_fuzzy_idx]['title'],
                temu_price=float(temu_product.get('price', 0)),
                amazon_price=float(amazon_products[best_fuzzy_idx].get('price', 0))
            )

        # Always try embedding comparison for top candidates
        temu_embedding = self.get_title_embedding(temu_title)
        best_match = None
        best_combined_score = 0

        # Check more candidates, even with lower fuzzy scores
        for idx, fuzzy_score in sorted_matches[:5]:  # Check top 5
            amazon_product = amazon_products[idx]
            amazon_embedding = self.get_title_embedding(amazon_product['title'])

            # Calculate embedding similarity
            embedding_score = self.cosine_similarity(temu_embedding, amazon_embedding)

            # Step 4: Image similarity (if available)
            image_score = 0.0
            if self.use_images:
                temu_img_path = temu_img_dir / f"{temu_id}.jpg"
                amazon_img_path = amazon_img_dir / f"{amazon_product['amazon_id']}.jpg"

                if temu_img_path.exists() and amazon_img_path.exists():
                    temu_img_emb = self.get_image_embedding(str(temu_img_path))
                    amazon_img_emb = self.get_image_embedding(str(amazon_img_path))

                    if temu_img_emb is not None and amazon_img_emb is not None:
                        image_score = self.cosine_similarity(temu_img_emb, amazon_img_emb)

            # Calculate combined score with adjusted weights
            if self.use_images and image_score > 0:
                # With images: fuzzy 25%, embedding 35%, image 40%
                combined_score = (fuzzy_score/100 * 0.25 + embedding_score * 0.35 + image_score * 0.40)
                method = "fuzzy+embedding+image"

                if embedding_score >= self.EMBEDDING_MIN and image_score >= self.IMAGE_MIN:
                    verdict = f"Strong match: fuzzy={fuzzy_score:.1f}%, embedding={embedding_score:.2f}, image={image_score:.2f}"
                    needs_review = False
                elif embedding_score >= self.EMBEDDING_MIN:
                    verdict = f"Good text match, moderate image: fuzzy={fuzzy_score:.1f}%, embedding={embedding_score:.2f}, image={image_score:.2f}"
                    needs_review = image_score < 0.7
                else:
                    verdict = f"Moderate match: fuzzy={fuzzy_score:.1f}%, embedding={embedding_score:.2f}, image={image_score:.2f}"
                    needs_review = True
            else:
                # Without images: fuzzy 40%, embedding 60%
                combined_score = (fuzzy_score/100 * 0.40 + embedding_score * 0.60)
                method = "fuzzy+embedding"

                if embedding_score >= self.EMBEDDING_MIN:
                    verdict = f"Good text match: fuzzy={fuzzy_score:.1f}%, embedding={embedding_score:.2f}"
                    needs_review = False
                else:
                    verdict = f"Moderate text match: fuzzy={fuzzy_score:.1f}%, embedding={embedding_score:.2f}"
                    needs_review = True

            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_match = MatchResult(
                    temu_id=temu_id,
                    amazon_id=amazon_product['amazon_id'],
                    confidence=combined_score,
                    match_method=method,
                    fuzzy_score=fuzzy_score,
                    embedding_score=embedding_score,
                    image_score=image_score,
                    verdict_reason=verdict,
                    needs_review=needs_review,
                    temu_title=temu_product['title'],
                    amazon_title=amazon_product['title'],
                    temu_price=float(temu_product.get('price', 0)),
                    amazon_price=float(amazon_product.get('price', 0))
                )

        # Always return the best match we found
        if best_match:
            return best_match
        else:
            # Fallback: return first match with low confidence
            return MatchResult(
                temu_id=temu_id,
                amazon_id=amazon_products[best_fuzzy_idx]['amazon_id'],
                confidence=best_fuzzy_score / 100 * 0.3,
                match_method="fuzzy_only",
                fuzzy_score=best_fuzzy_score,
                embedding_score=0.0,
                verdict_reason=f"Low fuzzy match ({best_fuzzy_score:.1f}%), needs manual review",
                needs_review=True,
                temu_title=temu_product['title'],
                amazon_title=amazon_products[best_fuzzy_idx]['title'],
                temu_price=float(temu_product.get('price', 0)),
                amazon_price=float(amazon_products[best_fuzzy_idx].get('price', 0))
            )

    def match_all_products(self, temu_file: str, amazon_file: str,
                          temu_img_dir: str = "temu_baby_toys_imgs",
                          amazon_img_dir: str = "amazon_baby_toys_imgs") -> List[MatchResult]:
        """Match all products between Temu and Amazon"""
        # Load data
        with open(temu_file, 'r', encoding='utf-8') as f:
            temu_data = json.load(f)

        with open(amazon_file, 'r', encoding='utf-8') as f:
            amazon_data = json.load(f)

        # Extract products
        temu_products = self._extract_products(temu_data, 'temu')
        amazon_products = self._extract_products(amazon_data, 'amazon')

        print(f"Loaded {len(temu_products)} Temu products and {len(amazon_products)} Amazon products")

        # Convert paths
        temu_img_path = Path(temu_img_dir)
        amazon_img_path = Path(amazon_img_dir)

        # Match each Temu product
        results = []
        for i, temu_product in enumerate(temu_products):
            if i % 10 == 0:
                print(f"Processing product {i+1}/{len(temu_products)}...")

            match = self.match_single_product(
                temu_product, amazon_products, temu_img_path, amazon_img_path
            )
            results.append(match)

        return results

    def _extract_products(self, data: Dict, source: str) -> List[Dict]:
        """Extract products from various JSON structures"""
        if source == 'temu':
            if 'all_products' in data:
                return data['all_products']
            elif 'products' in data:
                return data['products']
            elif 'results' in data:
                return data['results']
        else:  # amazon
            if 'all_products' in data:
                return data['all_products']
            elif 'products' in data:
                return data['products']
            elif 'top_prospects' in data:
                # Gather from different sections
                products = []
                for section in ['by_keyword_density', 'by_price_potential', 'by_margin_potential']:
                    if section in data['top_prospects']:
                        products.extend(data['top_prospects'][section])
                # Remove duplicates
                seen = set()
                unique = []
                for p in products:
                    if p['amazon_id'] not in seen:
                        seen.add(p['amazon_id'])
                        unique.append(p)
                return unique

        # Fallback
        return []

    def save_results(self, results: List[MatchResult], output_file: str = "matching_results.json"):
        """Save matching results to JSON"""
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_matches": len(results),
                "high_confidence": len([r for r in results if r.confidence >= 0.8]),
                "medium_confidence": len([r for r in results if 0.6 <= r.confidence < 0.8]),
                "low_confidence": len([r for r in results if r.confidence < 0.6]),
                "needs_review": len([r for r in results if r.needs_review]),
                "use_images": self.use_images
            },
            "matches": []
        }

        for result in results:
            output["matches"].append({
                "temu_id": result.temu_id,
                "amazon_id": result.amazon_id,
                "confidence": round(result.confidence, 3),
                # Add the missing fields here
                "temu_title": result.temu_title,
                "amazon_title": result.amazon_title,
                "temu_price": result.temu_price,
                "amazon_price": result.amazon_price,
                # Existing fields
                "match_method": result.match_method,
                "scores": {
                    "fuzzy": round(result.fuzzy_score, 1),
                    "embedding": round(result.embedding_score, 3),
                    "image": round(result.image_score, 3) if result.image_score > 0 else None
                },
                "verdict": result.verdict_reason,
                "needs_review": result.needs_review
            })

        # Sort by confidence
        output["matches"].sort(key=lambda x: x["confidence"], reverse=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {output_file}")
        print(f"High confidence matches (≥0.8): {output['metadata']['high_confidence']}")
        print(f"Medium confidence matches (0.6-0.8): {output['metadata']['medium_confidence']}")
        print(f"Low confidence matches (<0.6): {output['metadata']['low_confidence']}")
        print(f"Needs review: {output['metadata']['needs_review']}")

    def generate_llm_review_batch(
            self,
            results: List[MatchResult],
            temu_data: List[Dict],
            amazon_data: List[Dict],
            output_file: str = "llm_review_batch.json",
            confidence_threshold: float = 0.7
    ) -> None:
        """Generate batch of items for LLM review"""
        review_items = []

        for result in results:
            if result.needs_review or result.confidence < confidence_threshold:
                # Get full product data
                temu_product = next((p for p in temu_data if p.get('temu_id') == result.temu_id), None)
                amazon_product = next((p for p in amazon_data if p.get('amazon_id') == result.amazon_id), None)

                if temu_product and amazon_product:
                    review_items.append({
                        "temu": {
                            "temu_id": temu_product.get('temu_id'),
                            "title": temu_product.get('title'),
                            "price": temu_product.get('price'),
                            "image_available": Path(f"temu_baby_toys_imgs/{result.temu_id}.jpg").exists()
                        },
                        "amazon_candidate": {
                            "amazon_id": amazon_product.get('amazon_id'),
                            "title": amazon_product.get('title'),
                            "price": amazon_product.get('price'),
                            "keywords": amazon_product.get('high_value_keywords', []),
                            "image_available": Path(f"amazon_baby_toys_imgs/{result.amazon_id}.jpg").exists()
                        },
                        "match_scores": {
                            "fuzzy": round(result.fuzzy_score, 1),
                            "embedding": round(result.embedding_score, 3),
                            "image": round(result.image_score, 3) if result.image_score > 0 else None,
                            "confidence": round(result.confidence, 3)
                        },
                        "current_verdict": result.verdict_reason
                    })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"review_needed": review_items}, f, indent=2)

        print(f"\nGenerated {len(review_items)} items for LLM review in {output_file}")


def main():
    """Main execution"""
    # Initialize matcher
    print("Temu-Amazon Product Matcher")
    print("=" * 50)

    # Ask about image matching
    #use_images = input("\nUse image matching? (requires CLIP, y/n): ").lower() == 'y'

    matcher = ProductMatcher(use_images=True)  # skip the prompt

    # File paths
    temu_file = "temu_products_for_analysis.json"
    amazon_file = "amazon_products_for_analysis.json"

    # Check files exist
    if not Path(temu_file).exists():
        print(f"Error: {temu_file} not found!")
        return

    if not Path(amazon_file).exists():
        print(f"Error: {amazon_file} not found!")
        return

    # Run matching
    print("\nStarting product matching...")
    results = matcher.match_all_products(temu_file, amazon_file)

    # Save results
    matcher.save_results(results)

    # Generate LLM review batch if needed
    review_needed = sum(1 for r in results if r.needs_review)
    if review_needed > 0:
        print(f"\n{review_needed} matches need review. Generating LLM batch...")

        # Load full data for LLM batch
        with open(temu_file, 'r', encoding='utf-8') as f:
            temu_full = matcher._extract_products(json.load(f), 'temu')
        with open(amazon_file, 'r', encoding='utf-8') as f:
            amazon_full = matcher._extract_products(json.load(f), 'amazon')

        matcher.generate_llm_review_batch(results, temu_full, amazon_full)

    print("\nMatching complete!")


if __name__ == "__main__":
    main()
