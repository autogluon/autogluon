"""
Helper module for managing pycocotools dependency and providing utility functions.
Handles installation and import of pycocotools package, which has different
requirements on different platforms.
"""

import logging
import os
import tempfile
from types import ModuleType
from typing import Any, Optional

import portalocker

logger = logging.getLogger(__name__)


class PackageInstallError(Exception):
    """Exception raised when package installation fails."""

    pass


def import_try_install(package: str, extern_url: Optional[str] = None) -> ModuleType:
    """
    Try to import a package, installing it if necessary.

    Args:
        package: Name of the package to import
        extern_url: Optional external URL for package installation
                   (e.g., git repository URL)

    Returns:
        Imported package module

    Raises:
        PackageInstallError: If package installation fails
    """
    # Create a lock file to prevent concurrent installations
    lockfile = os.path.join(tempfile.gettempdir(), f"{package}_install.lck")

    with portalocker.Lock(lockfile):
        try:
            return __import__(package)
        except ImportError:
            logger.info("Package %s not found. Attempting installation...", package)

            try:
                # Get pip main function
                try:
                    from pip import main as pipmain
                except ImportError:
                    from pip._internal import main as pipmain

                    # Handle pip 19.3+ which returns ModuleType
                    if isinstance(pipmain, ModuleType):
                        from pip._internal.main import main as pipmain

                # Install package
                install_url = extern_url if extern_url else package
                result = pipmain(["install", "--user", install_url])

                if result != 0:
                    raise PackageInstallError(f"pip install failed with exit code {result}")

                # Try importing again
                try:
                    return __import__(package)
                except ImportError:
                    # Add user site packages to path if needed
                    import site
                    import sys

                    user_site = site.getusersitepackages()
                    if user_site not in sys.path:
                        sys.path.append(user_site)
                    return __import__(package)

            except Exception as e:
                raise PackageInstallError(f"Failed to install {package}: {str(e)}")


def try_import_pycocotools() -> None:
    """
    Try to import pycocotools, installing it if necessary.
    Handles platform-specific installation requirements.

    Raises:
        PackageInstallError: If pycocotools installation fails
    """
    try:
        import pycocotools

        logger.debug("pycocotools already installed")
        return
    except ImportError:
        logger.info("Installing pycocotools dependencies...")

        # Install Cython first (required for pycocotools)
        try:
            import_try_install("cython")
        except PackageInstallError as e:
            raise PackageInstallError(f"Failed to install Cython requirement: {str(e)}")

        # Install pycocotools with platform-specific handling
        try:
            if os.name == "nt":  # Windows
                # Use Windows-compatible fork
                win_url = "git+https://github.com/zhreshold/cocoapi.git#subdirectory=PythonAPI"
                import_try_install("pycocotools", win_url)
            else:  # Unix-like
                import_try_install("pycocotools")
        except PackageInstallError as e:
            raise PackageInstallError(
                "Failed to install pycocotools. Please check:\n"
                "1. You have gcc/g++ installed\n"
                "2. You have Python development files installed\n"
                "3. You have correct permissions\n"
                f"Error: {str(e)}"
            )


class COCOEvaluator:
    """
    Wrapper class for COCO evaluation functionality.
    Provides a more Pythonic interface to pycocotools evaluation.
    """

    def __init__(self):
        """Initialize COCO evaluator."""
        try_import_pycocotools()
        from pycocotools.cocoeval import COCOeval

        self.COCOeval = COCOeval

    def evaluate(self, gt_coco: Any, dt_coco: Any, eval_type: str = "bbox", verbose: bool = True) -> dict:
        """
        Perform COCO evaluation.

        Args:
            gt_coco: Ground truth COCO object
            dt_coco: Detection COCO object
            eval_type: Type of evaluation ('bbox', 'segm', 'keypoints')
            verbose: Whether to print evaluation progress

        Returns:
            Dictionary containing evaluation results
        """
        evaluator = self.COCOeval(gt_coco, dt_coco, eval_type)

        if not verbose:
            evaluator.params.verbose = 0

        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        return {"stats": evaluator.stats, "eval": evaluator.eval, "params": evaluator.params}


def verify_pycocotools_installation() -> bool:
    """
    Verify that pycocotools is properly installed and functional.

    Returns:
        True if installation is valid, False otherwise
    """
    try:
        import pycocotools
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        # Try to create dummy objects to verify functionality
        coco = COCO()
        eval = COCOeval(coco, coco, "bbox")

        logger.info("pycocotools installation verified successfully")
        return True

    except ImportError as e:
        logger.error("pycocotools import failed: %s", str(e))
        return False
    except Exception as e:
        logger.error("pycocotools verification failed: %s", str(e))
        return False


def get_pycocotools_version() -> Optional[str]:
    """
    Get the installed version of pycocotools.

    Returns:
        Version string if available, None if not installed
    """
    try:
        import pycocotools

        return getattr(pycocotools, "__version__", "unknown")
    except ImportError:
        return None
