import uuid
import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils.misc import shopee_dataset


def create_imbalanced_dataset(train_data, num_classes=4):
    """Create imbalanced dataset by downsampling each class progressively."""
    ds = 1
    imbalanced_data = []
    for lb in range(num_classes):
        class_data = train_data[train_data.label == lb]
        sample_index = np.random.choice(
            np.arange(len(class_data)), 
            size=int(len(class_data) * ds), 
            replace=False
        )
        ds /= 3
        imbalanced_data.append(class_data.iloc[sample_index])
    return pd.concat(imbalanced_data)


def compute_class_weights(data, num_classes=4):
    """Compute normalized inverse frequency weights for each class."""
    weights = []
    for lb in range(num_classes):
        class_data = data[data.label == lb]
        weights.append(1 / (class_data.shape[0] / data.shape[0]))
        print(f"Class {lb}: {len(class_data)} samples")
    weights = list(np.array(weights) / np.sum(weights))
    return weights


if __name__ == "__main__":
    # Prepare data
    download_dir = "./ag_automm_tutorial_imgcls_focalloss"
    train_data, test_data = shopee_dataset(download_dir)
    imbalanced_train_data = create_imbalanced_dataset(train_data)
    weights = compute_class_weights(imbalanced_train_data)
    print(f"Class weights: {weights}")

    # Train with focal loss
    predictor_focal = MultiModalPredictor(
        label="label",
        problem_type="multiclass",
        path=f"./tmp/{uuid.uuid4().hex}-automm_shopee_focal"
    )
    predictor_focal.fit(
        train_data=imbalanced_train_data,
        hyperparameters={
            "model.mmdet_image.checkpoint_name": "swin_tiny_patch4_window7_224",
            "env.num_gpus": 1,
            "optim.loss_func": "focal_loss",
            "optim.focal_loss.alpha": weights,
            "optim.focal_loss.gamma": 1.0,
            "optim.focal_loss.reduction": "sum",
            "optim.max_epochs": 10,
        },
    )
    focal_results = predictor_focal.evaluate(test_data, metrics=["acc"])

    # Train without focal loss
    predictor_baseline = MultiModalPredictor(
        label="label",
        problem_type="multiclass",
        path=f"./tmp/{uuid.uuid4().hex}-automm_shopee_non_focal"
    )
    predictor_baseline.fit(
        train_data=imbalanced_train_data,
        hyperparameters={
            "model.mmdet_image.checkpoint_name": "swin_tiny_patch4_window7_224",
            "env.num_gpus": 1,
            "optim.max_epochs": 10,
        },
    )
    baseline_results = predictor_baseline.evaluate(test_data, metrics=["acc"])

    # Compare results
    print("\n" + "=" * 50)
    print("FOCAL LOSS VERIFICATION RESULTS")
    print("=" * 50)
    print(f"Focal Loss Accuracy:    {focal_results}")
    print(f"Baseline Accuracy:      {baseline_results}")
