"""
Single Item Product Matcher
Matches a single user-provided product against all scraped Temu and Amazon products
Standalone version - does not depend on ProductMatcher class
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import pickle
import hashlib
import sys
from urllib.parse import urlparse
import requests
from io import BytesIO
from PIL import Image

# Core matching libraries
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import torch

# Force UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Optional: CLIP for image similarity
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    clip = None
    CLIP_AVAILABLE = False
    print("CLIP not installed. Image matching will be disabled.")
    print("Install with: pip install git+https://github.com/openai/CLIP.git")


@dataclass
class SingleItemMatch:
    """Represents a match between user item and a scraped product"""
    platform: str  # 'temu' or 'amazon'
    product_id: str
    title: str
    price: float
    confidence: float
    fuzzy_score: float
    embedding_score: float
    image_score: float
    match_method: str
    image_available: bool


class SingleItemMatcher:
    """Standalone matcher for comparing a single item against all products"""

    def __init__(self, use_images: bool = True, cache_dir: str = "embedding_cache"):
        """Initialize the single item matcher"""
        print("Initializing Single Item Matcher...")

        # Create cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Cache file paths
        self.text_cache_file = self.cache_dir / "text_embeddings.pkl"
        self.image_cache_file = self.cache_dir / "image_embeddings.pkl"
        self.metadata_cache_file = self.cache_dir / "cache_metadata.json"

        # Sentence transformer for semantic matching
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentence_model_name = 'all-MiniLM-L6-v2'

        # CLIP for image matching (if available and requested)
        self.use_images = use_images and CLIP_AVAILABLE
        self.clip_model_name = "ViT-B/32"

        if self.use_images:
            assert clip is not None
            self.clip_model, self.clip_preprocess = clip.load(
                self.clip_model_name,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            print("CLIP model loaded for image matching")

        # Load cached embeddings
        self.title_embeddings_cache = self._load_text_cache()
        self.image_embeddings_cache = self._load_image_cache()

        # Track if we've added new embeddings that need saving
        self.text_cache_dirty = False
        self.image_cache_dirty = False

        print("Single Item Matcher initialized")

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

    def _get_text_cache_key(self, text: str) -> str:
        """Generate a cache key for text"""
        cleaned = self.clean_title(text)
        return f"{cleaned[:100]}_{hashlib.md5(cleaned.encode()).hexdigest()[:8]}"

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

            # Save periodically (every 20 new embeddings)
            if len(self.image_embeddings_cache) % 20 == 0:
                self._save_caches()

            return embedding

        except Exception as e:
            print(f"Error processing image {image_path_or_url}: {e}")
            return None

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

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

    def load_all_products(self, category_name: str) -> Tuple[List[Dict], List[Dict]]:
        """Load products from both Temu and Amazon files"""
        # Construct file paths
        temu_file = Path(f"temu_{category_name}_analysis.json")
        amazon_file = Path(f"amazon_{category_name}_analysis.json")

        # Check files exist
        if not temu_file.exists():
            print(f"⚠️  Warning: {temu_file} not found!")
            temu_products = []
        else:
            with open(temu_file, 'r', encoding='utf-8') as f:
                temu_data = json.load(f)
            temu_products = self._extract_products(temu_data, 'temu')
            print(f"✓ Loaded {len(temu_products)} Temu products")

        if not amazon_file.exists():
            print(f"⚠️  Warning: {amazon_file} not found!")
            amazon_products = []
        else:
            with open(amazon_file, 'r', encoding='utf-8') as f:
                amazon_data = json.load(f)
            amazon_products = self._extract_products(amazon_data, 'amazon')
            print(f"✓ Loaded {len(amazon_products)} Amazon products")

        return temu_products, amazon_products

    def _is_url(self, string: str) -> bool:
        """Check if string is a URL"""
        try:
            result = urlparse(string)
            return all([result.scheme, result.netloc])
        except:
            return False

    def _validate_image_url(self, url: str) -> bool:
        """Check if image URL is accessible"""
        try:
            response = requests.head(url, timeout=5)
            return response.status_code == 200
        except:
            return False

    def prepare_user_item(self, title: str, image_path_or_url: str) -> Dict:
        """Prepare user item with embeddings"""
        # Validate image
        if self._is_url(image_path_or_url):
            image_type = "url"
            image_valid = self._validate_image_url(image_path_or_url)
        else:
            image_type = "file"
            image_valid = Path(image_path_or_url).exists()

        if not image_valid:
            print(f"⚠️  Warning: Image {'URL' if image_type == 'url' else 'file'} is not accessible")

        # Clean title
        title_clean = self.clean_title(title)

        # Get embeddings
        print("Computing embeddings for input item...")
        title_embedding = self.get_title_embedding(title)

        image_embedding = None
        if self.use_images and image_valid:
            image_embedding = self.get_image_embedding(image_path_or_url)

        return {
            'title': title,
            'title_clean': title_clean,
            'title_embedding': title_embedding,
            'image_path': image_path_or_url,
            'image_embedding': image_embedding,
            'image_valid': image_valid
        }

    def match_against_products(self, user_item: Dict, products: List[Dict],
                             platform: str, img_dir: Path) -> List[SingleItemMatch]:
        """Match user item against a list of products from one platform"""
        matches = []

        user_title_clean = user_item['title_clean']
        user_title_embedding = user_item['title_embedding']
        user_image_embedding = user_item['image_embedding']

        for product in products:
            # Get product ID based on platform
            product_id = product.get(f'{platform}_id', product.get('id', ''))
            product_title = product.get('title', '')
            product_price = float(product.get('price', 0))

            # Clean product title
            product_title_clean = self.clean_title(product_title)

            # Fuzzy matching
            fuzzy_score = max(
                fuzz.token_set_ratio(user_title_clean, product_title_clean),
                fuzz.partial_ratio(user_title_clean, product_title_clean)
            )

            # Skip if fuzzy score is too low
            if fuzzy_score < 30:
                continue

            # Embedding similarity
            product_embedding = self.get_title_embedding(product_title)
            embedding_score = self.cosine_similarity(user_title_embedding, product_embedding)

            # Image similarity
            image_score = 0.0
            image_available = False
            if self.use_images and user_image_embedding is not None:
                product_img_path = img_dir / f"{product_id}.jpg"
                if product_img_path.exists():
                    image_available = True
                    product_img_embedding = self.get_image_embedding(str(product_img_path))
                    if product_img_embedding is not None:
                        image_score = self.cosine_similarity(user_image_embedding, product_img_embedding)

            # Calculate combined confidence
            if self.use_images and image_score > 0:
                # With images: fuzzy 25%, embedding 35%, image 40%
                confidence = (fuzzy_score/100 * 0.25 + embedding_score * 0.35 + image_score * 0.40)
                method = "fuzzy+embedding+image"
            else:
                # Without images: fuzzy 40%, embedding 60%
                confidence = (fuzzy_score/100 * 0.40 + embedding_score * 0.60)
                method = "fuzzy+embedding"

            match = SingleItemMatch(
                platform=platform,
                product_id=product_id,
                title=product_title,
                price=product_price,
                confidence=confidence,
                fuzzy_score=fuzzy_score,
                embedding_score=embedding_score,
                image_score=image_score,
                match_method=method,
                image_available=image_available
            )

            matches.append(match)

        return matches

    def find_matches(self, title: str, image_path_or_url: str,
                    category_name: str, top_n: int = 20) -> Dict:
        """Find best matches for a single item across all products"""
        print(f"\nSearching for matches in category: {category_name}")
        print(f"Input title: {title}")
        print(f"Input image: {image_path_or_url}")

        # Prepare user item
        user_item = self.prepare_user_item(title, image_path_or_url)

        # Load all products
        temu_products, amazon_products = self.load_all_products(category_name)

        # Set up image directories
        temu_img_dir = Path(f"temu_{category_name}_imgs")
        amazon_img_dir = Path(f"amazon_{category_name}_imgs")

        # Match against both platforms
        all_matches = []

        if temu_products:
            print(f"\nMatching against {len(temu_products)} Temu products...")
            temu_matches = self.match_against_products(
                user_item, temu_products, 'temu', temu_img_dir
            )
            all_matches.extend(temu_matches)
            print(f"Found {len(temu_matches)} potential Temu matches")

        if amazon_products:
            print(f"\nMatching against {len(amazon_products)} Amazon products...")
            amazon_matches = self.match_against_products(
                user_item, amazon_products, 'amazon', amazon_img_dir
            )
            all_matches.extend(amazon_matches)
            print(f"Found {len(amazon_matches)} potential Amazon matches")

        # Sort by confidence
        all_matches.sort(key=lambda x: x.confidence, reverse=True)

        # Take top N
        top_matches = all_matches[:top_n]

        # Save caches
        self._save_caches()

        # Prepare results
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "category": category_name,
                "total_products_checked": len(temu_products) + len(amazon_products),
                "matches_found": len(all_matches),
                "top_n_returned": len(top_matches),
                "use_images": self.use_images
            },
            "input_item": {
                "title": user_item['title'],
                "image_path": user_item['image_path'],
                "image_valid": user_item['image_valid']
            },
            "matches": []
        }

        for match in top_matches:
            results["matches"].append({
                "platform": match.platform,
                "product_id": match.product_id,
                "title": match.title,
                "price": match.price,
                "confidence": round(match.confidence, 3),
                "match_method": match.match_method,
                "scores": {
                    "fuzzy": round(match.fuzzy_score, 1),
                    "embedding": round(match.embedding_score, 3),
                    "image": round(match.image_score, 3) if match.image_score > 0 else None
                },
                "image_available": match.image_available
            })

        return results

    def save_results(self, results: Dict, output_file: str = None):
        """Save matching results to JSON"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"single_item_matches_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_file}")

        # Print summary
        print(f"\nMatch Summary:")
        print(f"Total matches found: {len(results['matches'])}")

        if results['matches']:
            # Count by platform
            temu_count = sum(1 for m in results['matches'] if m['platform'] == 'temu')
            amazon_count = sum(1 for m in results['matches'] if m['platform'] == 'amazon')

            print(f"  - Temu: {temu_count}")
            print(f"  - Amazon: {amazon_count}")

            # Show top 5
            print(f"\nTop 5 matches:")
            for i, match in enumerate(results['matches'][:5], 1):
                print(f"{i}. [{match['platform'].upper()}] {match['title'][:60]}...")
                print(f"   Price: ${match['price']:.2f} | Confidence: {match['confidence']:.2%}")


def sanitize_filename(filename: str) -> str:
    """Sanitize string for use in filenames"""
    sanitized = re.sub(r'[^\w\s-]', '', filename)
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    return sanitized.lower()


def main():
    """Main execution"""
    print("Single Item Product Matcher")
    print("=" * 50)

    # Get product category
    print("\nEnter the product category to search in (e.g., 'baby toys', 'montessori_toys'):")
    category_input = input("Category name: ").strip()
    category_name = sanitize_filename(category_input)

    # Get user item details
    print("\nEnter the product title/name:")
    title = input("Title: ").strip()

    print("\nEnter the image path or URL:")
    print("Examples:")
    print("  - Local file: /path/to/image.jpg")
    print("  - URL: https://example.com/image.jpg")
    image_path = input("Image: ").strip()

    # Ask about image matching
    use_images = input("\nUse image matching? (requires CLIP, y/n): ").lower() == 'y'

    # Optional: number of results
    top_n_input = input("\nHow many top matches to return? (default: 20): ").strip()
    top_n = int(top_n_input) if top_n_input.isdigit() else 20

    # Initialize matcher
    matcher = SingleItemMatcher(use_images=use_images)

    # Find matches
    print("\nSearching for matches...")
    results = matcher.find_matches(
        title=title,
        image_path_or_url=image_path,
        category_name=category_name,
        top_n=top_n
    )

    # Save results
    output_file = f"single_item_matches_{category_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    matcher.save_results(results, output_file)

    # Ask if user wants to view results
    if input("\nView detailed results? (y/n): ").lower() == 'y':
        print("\nDetailed Results:")
        print("-" * 80)
        for i, match in enumerate(results['matches'], 1):
            print(f"\n{i}. [{match['platform'].upper()}] {match['title']}")
            print(f"   ID: {match['product_id']}")
            print(f"   Price: ${match['price']:.2f}")
            print(f"   Confidence: {match['confidence']:.2%}")
            print(f"   Scores - Fuzzy: {match['scores']['fuzzy']}%, "
                  f"Embedding: {match['scores']['embedding']:.3f}", end="")
            if match['scores']['image'] is not None:
                print(f", Image: {match['scores']['image']:.3f}")
            else:
                print()
            print(f"   Image available: {'Yes' if match['image_available'] else 'No'}")


if __name__ == "__main__":
    main()