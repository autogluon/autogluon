"""Fine-tuning defaults, independent of checkpoint release identity."""

DEFAULT_FINE_TUNE_RECIPE = "2"

FINETUNE_RECIPES = {
    "1": {
        "lr": 1e-4,
        "warmup_steps": 1000,
        "weight_decay": 0.1,
        "max_samples_support": 8192,
        "max_samples_query": 1024,
        "predict_max_samples_support": 8192,
        "predict_max_samples_query": 1024,
        "eval_max_samples_query": 1024,
        "balance_binary_support": False,
        "heldout_in_support": False,
        "fast_loop": False,
        "fast_predict": False,
        "feature_reduction": None,
    },
    "2": {
        "lr": None,  # Data-dependent: 3e-6 for small binary tables, otherwise 1e-5.
        "warmup_steps": 10,
        "weight_decay": 0.3,
        "max_samples_support": None,  # Classification: 16384; regression: 20480.
        "max_samples_query": 1024,
        "predict_max_samples_support": None,  # Binary: 16384; otherwise: 32768.
        "predict_max_samples_query": 16384,
        "eval_max_samples_query": 16384,
        "balance_binary_support": True,
        "heldout_in_support": True,
        "fast_loop": True,
        "fast_predict": True,
        "feature_reduction": "auto",
    },
}


def get_finetune_recipe(recipe: str) -> dict:
    if not isinstance(recipe, str) or recipe not in FINETUNE_RECIPES:
        raise ValueError(f"Unknown fine_tune_recipe={recipe!r}. Expected '1' or '2'.")
    return FINETUNE_RECIPES[recipe]
