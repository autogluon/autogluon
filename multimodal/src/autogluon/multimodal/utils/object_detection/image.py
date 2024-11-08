"""
Image utilities for object detection.
Provides functions for image loading, processing, and related operations.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Union
from urllib.parse import urlparse

import PIL.Image
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _get_image_info(image_path: str) -> Optional[Dict[str, int]]:
    """
    Get image dimensions without loading the entire image into memory.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary containing 'height' and 'width' if successful, None otherwise
    """
    try:
        with PIL.Image.open(image_path) as img:
            width, height = img.size
            return {
                "height": height,
                "width": width
            }
    except Exception as e:
        warnings.warn(f"Could not read image {image_path}: {str(e)}")
        return None


def get_image_filename(path: str) -> str:
    """
    Extract the filename without extension from an image path.

    Args:
        path: Path to the image file

    Returns:
        Filename without extension
    """
    return Path(path.replace("\\", "/")).stem


def is_url(resource_path: str) -> bool:
    """
    Check if a given path is a URL.

    Args:
        resource_path: Path to check

    Returns:
        True if path is a URL, False otherwise
    """
    try:
        result = urlparse(resource_path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def download(url: str, save_dir: Optional[str] = None) -> str:
    """
    Download a file from a URL with progress bar.

    Args:
        url: URL to download from
        save_dir: Optional directory to save the file to.
                 If None, saves to temporary directory.

    Returns:
        Path to downloaded file

    Raises:
        ValueError: If URL is invalid or download fails
        IOError: If file cannot be saved
    """
    if not is_url(url):
        raise ValueError(f"Invalid URL: {url}")

    if save_dir is None:
        import tempfile
        save_dir = tempfile.gettempdir()

    try:
        # Extract filename from URL
        filename = os.path.basename(urlparse(url).path)
        if not filename:
            filename = 'downloaded_file'

        save_path = os.path.join(save_dir, filename)
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB

        with open(save_path, 'wb') as f, tqdm(
            desc=f"Downloading {filename}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)

        logger.info(f"Downloaded file saved to: {save_path}")
        return save_path

    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to download from {url}: {str(e)}")
    except IOError as e:
        raise IOError(f"Failed to save downloaded file: {str(e)}")


class ImageLoader:
    """Utility class for efficient image loading and caching."""
    
    def __init__(self, cache_size: int = 100):
        """
        Initialize ImageLoader.

        Args:
            cache_size: Maximum number of images to keep in memory
        """
        self.cache_size = cache_size
        self.cache = {}
        self._access_count = 0

    def load_image(self, image_path: str) -> Optional[PIL.Image.Image]:
        """
        Load an image, using cache if available.

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image if successful, None otherwise
        """
        # Return cached image if available
        if image_path in self.cache:
            self._access_count += 1
            self.cache[image_path]['last_access'] = self._access_count
            return self.cache[image_path]['image']

        # Load new image
        try:
            image = PIL.Image.open(image_path)
            
            # Manage cache size
            if len(self.cache) >= self.cache_size:
                # Remove least recently used image
                lru_path = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k]['last_access']
                )
                del self.cache[lru_path]

            # Cache new image
            self._access_count += 1
            self.cache[image_path] = {
                'image': image,
                'last_access': self._access_count
            }
            
            return image

        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {str(e)}")
            return None

    def clear_cache(self):
        """Clear the image cache."""
        self.cache.clear()
        self._access_count = 0


def verify_image_directory(
    directory: Union[str, Path],
    extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')
) -> tuple[list, list]:
    """
    Verify all images in a directory are readable and have valid dimensions.

    Args:
        directory: Path to directory containing images
        extensions: Tuple of valid image extensions

    Returns:
        Tuple containing:
            - List of valid image paths
            - List of invalid image paths
    """
    if isinstance(directory, str):
        directory = Path(directory)

    valid_images = []
    invalid_images = []
    
    # Check if directory exists
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    # Iterate through all files in directory
    for filepath in tqdm(
        list(directory.glob("*")),
        desc="Verifying images",
        unit="files"
    ):
        if filepath.suffix.lower() in extensions:
            image_info = _get_image_info(str(filepath))
            
            if image_info is not None and image_info['width'] > 0 and image_info['height'] > 0:
                valid_images.append(str(filepath))
            else:
                invalid_images.append(str(filepath))
                logger.warning(f"Invalid image found: {filepath}")

    # Log summary
    logger.info(
        f"Image verification complete: "
        f"{len(valid_images)} valid, {len(invalid_images)} invalid"
    )

    return valid_images, invalid_images


def create_image_thumbnail(
    image_path: str,
    max_size: tuple = (128, 128),
    output_path: Optional[str] = None
) -> Optional[PIL.Image.Image]:
    """
    Create a thumbnail of an image while maintaining aspect ratio.

    Args:
        image_path: Path to input image
        max_size: Maximum (width, height) of thumbnail
        output_path: Optional path to save thumbnail

    Returns:
        PIL Image thumbnail if successful, None otherwise
    """
    try:
        with PIL.Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Create thumbnail
            img.thumbnail(max_size, PIL.Image.LANCZOS)
            
            if output_path:
                img.save(output_path, quality=95, optimize=True)
            
            return img

    except Exception as e:
        logger.error(f"Failed to create thumbnail for {image_path}: {str(e)}")
        return None
