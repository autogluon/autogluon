#!/usr/bin/env python3
"""
Script to cache HuggingFace models from S3 before AutoGluon training starts.
This prevents rate limiting issues when multiple parallel benchmark runs try to download TabPFNv2 models.
"""

import subprocess
from pathlib import Path


def setup_hf_cache():
    """Download tabular foundational models from S3 to HuggingFace cache directory."""

    # HuggingFace cache directory
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # S3 bucket with model mirror
    bucket = "s3://autogluon-hf-model-mirror"

    # tabular foundational models to cache
    models = [
        "Prior-Labs/TabPFN-v2-clf",
        "Prior-Labs/TabPFN-v2-reg"
    ]

    print("Setting up HuggingFace model cache for tabular foundational models...")

    for model in models:
        # Convert model name to S3 path format
        model_name = "--".join(model.split('/'))
        s3_model_name = f"models--{model_name}"
        model_path = cache_dir / s3_model_name
        s3_path = f"{bucket}/{s3_model_name}"

        print(f"Caching {model} from {s3_path}...")

        try:
            # Create local directory
            model_path.mkdir(parents=True, exist_ok=True)

            # Download from S3 to local cache
            cmd = ["aws", "s3", "cp", s3_path, str(model_path), "--recursive", "--quiet"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"Successfully cached {model}")
            else:
                print(f"Warning: Failed to cache {model} from S3: {result.stderr}")
                print("Tabular foundational models will be downloaded from HuggingFace Hub during training")

        except Exception as e:
            print(f"Warning: Exception while caching {model}: {e}")
            print("Tabular foundational models will be downloaded from HuggingFace Hub during training")

    print("HuggingFace model cache setup completed.")


if __name__ == "__main__":
    setup_hf_cache()
