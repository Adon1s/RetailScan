"""
Temu-Amazon Product Matcher with Persistent Embedding Cache
Matches products between Temu and Amazon using multiple similarity methods
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import pickle
import hashlib

# Core matching libraries
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
import requests
from io import BytesIO
import sys

# Force UTF-8 output so Windows console won't choke on ✓, ✗, etc.
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
    def __init__(self, use_images: bool = True, cache_dir: str = "embedding_cache"):
        """Initialize the matcher with models and load cached embeddings"""
        print("Initializing ProductMatcher...")

        # Create cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Cache file paths
        self.text_cache_file = self.cache_dir / "text_embeddings.pkl"
        self.image_cache_file = self.cache_dir / "image_embeddings.pkl"
        self.metadata_cache_file = self.cache_dir / "cache_metadata.json"

        # Sentence transformer for semantic matching
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentence_model_name = 'all-MiniLM-L6-v2'  # Store for cache validation

        # CLIP for image matching (if available and requested)
        print(f"  › use_images flag: {use_images}, CLIP_AVAILABLE: {CLIP_AVAILABLE}")
        self.use_images = use_images and CLIP_AVAILABLE
        self.clip_model_name = "ViT-B/32"  # Store for cache validation

        if self.use_images:
            assert clip is not None  # Tell IDE that clip cannot be None here
            self.clip_model, self.clip_preprocess = clip.load(self.clip_model_name,
                                                             device="cuda" if torch.cuda.is_available() else "cpu")
            print("CLIP model loaded for image matching")

        # Load cached embeddings
        self.title_embeddings_cache = self._load_text_cache()
        self.image_embeddings_cache = self._load_image_cache()

        # Track if we've added new embeddings that need saving
        self.text_cache_dirty = False
        self.image_cache_dirty = False

        # Updated thresholds for better matching
        self.FUZZY_IMMEDIATE_ACCEPT = 90  # Lowered from 93
        self.FUZZY_CANDIDATE_MIN = 40     # Lowered significantly from 85
        self.EMBEDDING_MIN = 0.70          # Lowered from 0.80
        self.IMAGE_MIN = 0.85              # Lowered from 0.88

    def _get_cache_metadata(self):
        """Get current cache metadata for validation"""
        return {
            "sentence_model": self.sentence_model_name,
            "clip_model": self.clip_model_name if self.use_images else None,
            "cache_version": "1.0"
        }

    def _validate_cache_metadata(self, stored_metadata):
        """Check if cached embeddings are compatible with current models"""
        current = self._get_cache_metadata()
        return (stored_metadata.get("sentence_model") == current["sentence_model"] and
                stored_metadata.get("clip_model") == current["clip_model"] and
                stored_metadata.get("cache_version") == current["cache_version"])

    def _load_text_cache(self):
        """Load text embeddings from cache if valid"""
        if not self.text_cache_file.exists():
            print("  › No text embedding cache found, starting fresh")
            return {}

        try:
            # Check metadata first
            if self.metadata_cache_file.exists():
                with open(self.metadata_cache_file, 'r') as f:
                    metadata = json.load(f)
                if not self._validate_cache_metadata(metadata):
                    print("  › Text cache outdated (model changed), starting fresh")
                    return {}

            with open(self.text_cache_file, 'rb') as f:
                cache = pickle.load(f)
            print(f"  › Loaded {len(cache)} cached text embeddings")
            return cache
        except Exception as e:
            print(f"  › Error loading text cache: {e}, starting fresh")
            return {}

    def _load_image_cache(self):
        """Load image embeddings from cache if valid"""
        if not self.use_images or not self.image_cache_file.exists():
            return {}

        try:
            # Check metadata first
            if self.metadata_cache_file.exists():
                with open(self.metadata_cache_file, 'r') as f:
                    metadata = json.load(f)
                if not self._validate_cache_metadata(metadata):
                    print("  › Image cache outdated (model changed), starting fresh")
                    return {}

            with open(self.image_cache_file, 'rb') as f:
                cache = pickle.load(f)
            print(f"  › Loaded {len(cache)} cached image embeddings")
            return cache
        except Exception as e:
            print(f"  › Error loading image cache: {e}, starting fresh")
            return {}

    def _save_caches(self):
        """Save any dirty caches to disk"""
        if self.text_cache_dirty:
            try:
                with open(self.text_cache_file, 'wb') as f:
                    pickle.dump(self.title_embeddings_cache, f)
                print(f"  › Saved {len(self.title_embeddings_cache)} text embeddings to cache")
                self.text_cache_dirty = False
            except Exception as e:
                print(f"  › Error saving text cache: {e}")

        if self.image_cache_dirty and self.use_images:
            try:
                with open(self.image_cache_file, 'wb') as f:
                    pickle.dump(self.image_embeddings_cache, f)
                print(f"  › Saved {len(self.image_embeddings_cache)} image embeddings to cache")
                self.image_cache_dirty = False
            except Exception as e:
                print(f"  › Error saving image cache: {e}")

        # Save metadata
        if self.text_cache_dirty or self.image_cache_dirty:
            try:
                with open(self.metadata_cache_file, 'w') as f:
                    json.dump(self._get_cache_metadata(), f)
            except Exception as e:
                print(f"  › Error saving cache metadata: {e}")

    def _get_text_cache_key(self, text: str) -> str:
        """Generate a cache key for text"""
        # Use cleaned text for consistency
        cleaned = self.clean_title(text)
        # Add a hash for safety (in case of very long titles)
        return f"{cleaned[:100]}_{hashlib.md5(cleaned.encode()).hexdigest()[:8]}"

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
        cache_key = self._get_text_cache_key(title)

        if cache_key not in self.title_embeddings_cache:
            # Compute embedding
            embedding = self.sentence_model.encode(title)
            self.title_embeddings_cache[cache_key] = embedding
            self.text_cache_dirty = True

            # Save periodically (every 50 new embeddings)
            if len(self.title_embeddings_cache) % 50 == 0:
                self._save_caches()

        return self.title_embeddings_cache[cache_key]

    def get_image_embedding(self, image_path_or_url: str) -> Optional[np.ndarray]:
        """Get CLIP embedding for image (cached)"""
        if not self.use_images:
            return None

        # Use the full path/URL as cache key for images
        cache_key = str(image_path_or_url)

        if cache_key in self.image_embeddings_cache:
            return self.image_embeddings_cache[cache_key]

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

            self.image_embeddings_cache[cache_key] = embedding
            self.image_cache_dirty = True

            # Save periodically (every 20 new embeddings - images are more expensive)
            if len(self.image_embeddings_cache) % 20 == 0:
                self._save_caches()

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

        # Save any remaining cached embeddings
        self._save_caches()

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
                "use_images": self.use_images,
                "cache_stats": {
                    "text_embeddings_cached": len(self.title_embeddings_cache),
                    "image_embeddings_cached": len(self.image_embeddings_cache)
                }
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
        print(f"Embeddings cached - Text: {output['metadata']['cache_stats']['text_embeddings_cached']}, "
              f"Images: {output['metadata']['cache_stats']['image_embeddings_cached']}")

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

    def clear_cache(self, text_only=False, image_only=False):
        """Clear embedding caches"""
        if not text_only and not image_only:
            # Clear both
            self.title_embeddings_cache = {}
            self.image_embeddings_cache = {}
            print("Cleared all embedding caches")
        elif text_only:
            self.title_embeddings_cache = {}
            print("Cleared text embedding cache")
        elif image_only:
            self.image_embeddings_cache = {}
            print("Cleared image embedding cache")

        # Mark as dirty to force save
        self.text_cache_dirty = True
        self.image_cache_dirty = True
        self._save_caches()


def main():
    """Main execution"""
    # Initialize matcher
    print("Temu-Amazon Product Matcher")
    print("=" * 50)

    # Get the file name part from user
    print("\nEnter the product category (e.g., 'baby_toys', 'montessori_toys'):")
    print("This will look for analysis files named:")
    print("  - temu_[category]_analysis.json")
    print("  - amazon_[category]_analysis.json")

    category_name = input("Category name: ").strip()

    # Construct file paths
    temu_file = Path(f"temu_{category_name}_analysis.json")
    amazon_file = Path(f"amazon_{category_name}_analysis.json")
    temu_img_dir = f"temu_{category_name}_imgs"
    amazon_img_dir = f"amazon_{category_name}_imgs"

    # Check files exist
    if not temu_file.exists():
        print(f"\n❌ Error: {temu_file} not found!")
        print(f"   Make sure you've run the Temu data organizer for '{category_name}' first.")
        return

    if not amazon_file.exists():
        print(f"\n❌ Error: {amazon_file} not found!")
        print(f"   Make sure you've run the Amazon data organizer for '{category_name}' first.")
        return

    # Check image directories (warning only)
    if not Path(temu_img_dir).exists():
        print(f"\n⚠️  Warning: Temu images directory '{temu_img_dir}' not found.")
    if not Path(amazon_img_dir).exists():
        print(f"\n⚠️  Warning: Amazon images directory '{amazon_img_dir}' not found.")

    print(f"\n✓ Found Temu file: {temu_file}")
    print(f"✓ Found Amazon file: {amazon_file}")

    # Ask about image matching
    # use_images = input("\nUse image matching? (requires CLIP, y/n): ").lower() == 'y'

    matcher = ProductMatcher(use_images=True)  # skip the prompt

    # Run matching
    print("\nStarting product matching...")
    results = matcher.match_all_products(
        str(temu_file),
        str(amazon_file),
        temu_img_dir=temu_img_dir,
        amazon_img_dir=amazon_img_dir
    )

    # Save results with category-specific filename
    output_file = f"matching_results_{category_name}.json"
    matcher.save_results(results, output_file)

    # Generate LLM review batch if needed
    review_needed = sum(1 for r in results if r.needs_review)
    if review_needed > 0:
        print(f"\n{review_needed} matches need review. Generating LLM batch...")

        # Load full data for LLM batch
        with open(temu_file, 'r', encoding='utf-8') as f:
            temu_full = matcher._extract_products(json.load(f), 'temu')
        with open(amazon_file, 'r', encoding='utf-8') as f:
            amazon_full = matcher._extract_products(json.load(f), 'amazon')

        review_file = f"llm_review_batch_{category_name}.json"
        matcher.generate_llm_review_batch(results, temu_full, amazon_full, review_file)

    print("\nMatching complete!")


if __name__ == "__main__":
    main()