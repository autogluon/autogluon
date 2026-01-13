"""
The derivative application of Tip-Adapter (https://arxiv.org/pdf/2207.09519.pdf).
Refer to https://github.com/gaopengcuhk/Tip-Adapter
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from utils import *

from autogluon.multimodal import MultiModalPredictor


def get_args():
    parser = argparse.ArgumentParser(description="MultiModal Fewshot training with memory bank.")
    parser.add_argument(
        "--type",
        type=str,
        default="clip",
        help="Choose type of few-shot learning from 'clip', 'text', 'image' for different backbone methods.",
    )
    parser.add_argument("--backbone", type=str, default=None, help="The backbone model of MultiModal Predictor.")
    parser.add_argument("--data_path", type=str, default="./data/", help="The path for image dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="food101",
        help="The name of dataset. Support image datasets in COOP and text datasets in Huggingface.",
    )
    parser.add_argument("--column_names", nargs="+", default=None, help="The name of the data column.")
    parser.add_argument("--label_column", type=str, default="label", help="The name of the label column.")
    parser.add_argument("--shots", type=int, default=16, help="The shots for each class in training set.")
    parser.add_argument("--aug_epochs", type=int, default=1, help="The epochs to create the bank.")
    parser.add_argument(
        "--model_head_type",
        type=str,
        default="linear",
        help="The model head for few-shot classification. Choose from 'linear' and 'SVM'.",
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="The learning rate for training the model head.")
    parser.add_argument("--lr_F", type=float, default=1e-3, help="The learning rate for finetuing the memory bank.")
    parser.add_argument("--train_epoch", type=int, default=20, help="The training epochs for training the model head.")
    parser.add_argument(
        "--train_epoch_F", type=int, default=20, help="The training epochs for finetuning the memory bank."
    )
    parser.add_argument(
        "--init_alpha",
        type=float,
        default=1.17,
        help="The initial value of hyper-parameter alpha in memory bank. Alpha adjusts the weight of probability between the classifier and memory bank.",
    )
    parser.add_argument(
        "--init_beta",
        type=float,
        default=1.0,
        help="The initial values of hyper-parameter beta in memory bank. Beta modulates the sharpness when converting the similarities into non-negative values.",
    )
    parser.add_argument(
        "--search_scale", type=int, nargs="+", default=[10, 10], help="The search scale of alpha and beta."
    )
    parser.add_argument(
        "--search_step",
        type=int,
        nargs="+",
        default=[200, 20],
        help="The steps of searching hyper-parameters alpha and beta.",
    )
    args = parser.parse_args()

    args.bank_dir = os.path.join("./memory_bank_models", args.dataset)
    os.makedirs(args.bank_dir, exist_ok=True)

    if args.column_names is None:
        args.column_names = ["text"] if args.type == "text" else ["image"]

    if args.backbone is None:
        if args.type == "text":
            args.backbone = "sentence-transformers/paraphrase-mpnet-base-v2"
        elif args.type == "image":
            args.backbone = "swin_base_patch4_window7_224"
        else:
            args.backbone = "openai/clip-vit-large-patch14-336"

    return args


class AutoMMMemoryBank(nn.Module):
    """
    The model to generate few shot predict probability with
    features extracted by AutoGluon MultiModal Predictor.
    """

    def __init__(
        self,
        bank_keys,
        bank_labels,
        hidden_size,
        num_classes,
        clip_weights=None,
        model_head_type="linear",
    ):
        """
        Create the model head and the memory bank.

        Parameters
        ----------
        bank_keys
            The content of bank composed of features in the training set.
        bank_labels
            The labels of corresponding bank_keys.
        hidden_size
            The size of features.
        num_classes
            The classes of the dataset.
        clip_weights
            The clip embedding of the semantic text that describes the labels.
        model_head_type
            The type of the few-shot classification head.
        """
        super(AutoMMMemoryBank, self).__init__()
        self.bank_keys = bank_keys
        self.bank_values = F.one_hot(bank_labels).float()
        self.adapter = nn.Linear(bank_keys.shape[0], bank_keys.shape[1], bias=False)
        self.adapter.weight = nn.Parameter(bank_keys.t())

        self.clip_weights = clip_weights

        self.model_head_type = model_head_type
        if clip_weights is None:
            if model_head_type == "SVM":
                self.model_head = make_pipeline(StandardScaler(), SVC(gamma="auto", probability=True))
                self.model_head.fit(bank_keys.t().cpu(), bank_labels.cpu())
            else:
                self.model_head = nn.Linear(hidden_size, num_classes, bias=True) if clip_weights is None else None
        else:
            self.model_head = None

    def adapt_logits(self, affinity, pure_logits, alpha, beta):
        """
        Generate logits with memory bank based on pure_logits and bank output.

        Parameters
        ----------
        affinity
            The result of bank similarity. It is based on cosine similarity or a projector initialized with bank_keys.
        pure_logits
            The predict probability of the classifier.
        alpha
            The hyper-parameters of bank model.
        beta
            The hyper-parameters of bank model.

        Return
        ------
            The logits with memory bank.
        """
        bank_logits = ((-1) * (beta - beta * affinity)).exp() @ self.bank_values
        logits = pure_logits + bank_logits * alpha
        return logits

    def change_head_state(self, grad_state):
        """
        Change the training state of the model head.

        Parameters
        ----------
        grad_state
            The training state of the model head. If "True", the model head is trainable. If "False", the model head is freezed.
        """
        if self.model_head is not None and self.model_head_type == "linear":
            for param in self.model_head.parameters():
                param.requires_grad = grad_state

    def change_adapter_state(self, grad_state):
        """
        Change the training state of the memory bank.

        Parameters
        ----------
        grad_state
            The training state of the memory bank. If "True", the memory bank is trainable. If "False", the memory bank is freezed.
        """
        for param in self.adapter.parameters():
            param.requires_grad = grad_state

    def forward(self, x, alpha=1, beta=1, pure_logits=None):
        """
        Generate three types of logits with features.

        Parameters
        ----------
        x
            The image/text features generated by AutoGluon Multimodal Predictor.
        alpha
            The hyper-parameters of memory bank model.
        beta
            The hyper-parameters of memory bank model.

        Return
        ------
        The predict probability of the feature.
            - "pure_logits"
                The predict probability of classifier.
            - "adapted_logits"
                The predict probability composed of classifier and memory bank similarity result.
            - "adapted_logits_with_finetuning"
                The predict probability composed of classifier and fine-tuned memory bank result.
        """
        if pure_logits is None:
            if self.clip_weights is not None:
                pure_logits = 100.0 * x @ self.clip_weights
            elif self.model_head_type == "SVM":
                pure_logits = torch.tensor(self.model_head.predict_proba(x.cpu())).cuda()
            else:
                pure_logits = self.model_head(x)

        affinity = x @ self.bank_keys
        adapted_logits = self.adapt_logits(affinity, pure_logits, alpha, beta)

        finetuned_affinity = self.adapter(x)
        adapted_logits_with_finetuning = self.adapt_logits(finetuned_affinity, pure_logits, alpha, beta)

        return {
            "pure_logits": pure_logits,
            "adapted_logits": adapted_logits,
            "adapted_logits_with_finetuning": adapted_logits_with_finetuning,
        }


def train_logits(
    args,
    val_features,
    val_labels,
    test_features,
    test_labels,
    predictor,
    memory_bank_model,
    alpha,
    beta,
    loader,
    logits_type="pure_logits",
):
    """
    The training process of AutoMMMemoryBank.

    Parameters
    ----------
    args
        The args of the training process.
    val_features, val_labels
        The preprocessed features and labels of validation set.
    test_features, test_labels
        The preprocessed features and labels of test set.
    predictor
        The AutoGluon MultiModal Predictor to extract the features.
    memory_bank_model
        The AutoMMMemoryBank to generate logits and logits with memory bank.
    alpha, beta
        The hyper-parameters of memory bank.
    loader
        The dataset loader for training.
    logits_type
        The target logits of training corresponding to "pure_logits", "adapted_logits", "adapted_logits_with_finetuning".

    Return
    ------
    The best model after training.
    """
    if logits_type != "adapted_logits_with_finetuning":
        memory_bank_model.change_head_state(grad_state=True)
        memory_bank_model.change_adapter_state(grad_state=False)
        lr, train_epoch = args.lr, args.train_epoch
    else:
        memory_bank_model.change_head_state(grad_state=False)
        memory_bank_model.change_adapter_state(grad_state=True)
        lr, train_epoch = args.lr_F, args.train_epoch_F

    optimizer = torch.optim.AdamW(memory_bank_model.parameters(), lr=lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epoch * len(loader))
    memory_bank_model.train()

    best_acc, best_epoch = 0.0, 0
    for train_idx in range(train_epoch):
        correct_samples, all_samples = 0, 0
        loss_list = []
        print("Train Epoch: {:} / {:}".format(train_idx, train_epoch))

        for i, (data, target) in enumerate(tqdm(loader)):
            target = target.cuda()
            with torch.no_grad():
                features = extract_embedding(args, data, predictor, args.column_names)
                features /= features.norm(dim=-1, keepdim=True)

            logits = memory_bank_model(features, alpha, beta)

            loss = F.cross_entropy(logits[logits_type], target)

            acc = cls_acc(logits[logits_type], target)
            correct_samples += acc / 100 * len(logits[logits_type])
            all_samples += len(logits[logits_type])
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(
            "LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}".format(
                current_lr,
                correct_samples / all_samples,
                correct_samples,
                all_samples,
                sum(loss_list) / len(loss_list),
            )
        )

        memory_bank_model.eval()
        logits = memory_bank_model(val_features, alpha, beta)[logits_type]
        acc = cls_acc(logits, val_labels)
        print("**** val accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(memory_bank_model.state_dict(), args.bank_dir + "/best_F_" + str(args.shots) + "shots.pt")  # nosec B614

    memory_bank_model.load_state_dict(torch.load(args.bank_dir + "/best_F_" + str(args.shots) + "shots.pt"))  # nosec B614
    print(f"**** After fine-tuning {logits_type}, best val accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    return memory_bank_model


def run_memory_bank(
    args,
    val_features,
    val_labels,
    test_features,
    test_labels,
    predictor,
    memory_bank_model,
    loader,
):
    """
    Test the effectiveness of memory bank.
    Compare the results of pure_logits, adapted_logits and adapted_logits_with_finetuning.

    Parameters
    ----------
    args
        The args of the training process.
    val_features, val_labels
        The preprocessed features and labels of validation set.
    test_features, test_labels
        The preprocessed features and labels of test set.
    predictor
        The AutoGluon MultiModal Predictor to extract the features.
    memory_bank_model
        The AutoMMMemoryBank to generate logits and logits with memory bank.
    loader
        The dataset loader for training.
    """
    beta, alpha = args.init_beta, args.init_alpha

    if args.type != "clip" and args.model_head_type == "linear":
        memory_bank_model = train_logits(
            args=args,
            val_features=val_features,
            val_labels=val_labels,
            test_features=test_features,
            test_labels=test_labels,
            predictor=predictor,
            memory_bank_model=memory_bank_model,
            alpha=alpha,
            beta=beta,
            loader=loader,
            logits_type="pure_logits",
        )

    with torch.no_grad():
        logits = memory_bank_model(val_features, alpha, beta)
    acc = cls_acc(logits["pure_logits"], val_labels)
    print("\n**** Pure model's val accuracy: {:.2f}. ****\n".format(acc))

    acc = cls_acc(logits["adapted_logits"], val_labels)
    print("**** Model with memory_bank's val accuracy: {:.2f}. ****\n".format(acc))

    best_beta, best_alpha = search_hp(
        args=args,
        features=val_features,
        labels=val_labels,
        memory_bank_model=memory_bank_model,
        logits_type="adapted_logits",
    )

    print("\n-------- Evaluating on the test set. --------")

    with torch.no_grad():
        logits = memory_bank_model(test_features, best_alpha, best_beta)
    acc = cls_acc(logits["pure_logits"], test_labels)
    print("\n**** Pure Model's test accuracy: {:.2f}. ****\n".format(acc))

    acc = cls_acc(logits["adapted_logits"], test_labels)
    print("**** Model with Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    print("\n-------- Finetune Adapter. --------")
    memory_bank_model = train_logits(
        args=args,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_features,
        test_labels=test_labels,
        predictor=predictor,
        memory_bank_model=memory_bank_model,
        alpha=alpha,
        beta=beta,
        loader=loader,
        logits_type="adapted_logits_with_finetuning",
    )
    best_beta, best_alpha = search_hp(
        args=args,
        features=val_features,
        labels=val_labels,
        memory_bank_model=memory_bank_model,
        logits_type="adapted_logits_with_finetuning",
    )
    with torch.no_grad():
        logits = memory_bank_model(test_features, best_alpha, best_beta)
    acc = cls_acc(logits["adapted_logits_with_finetuning"], test_labels)
    print("**** Model with Finetuned Adapter's test accuracy: {:.2f}. ****\n".format(acc))


def main():
    dataset, train_df, val_df, test_df, num_classes = generate_dataset(args)

    if args.type == "text":
        predictor = MultiModalPredictor(
            problem_type="feature_extraction",
            hyperparameters={
                "model.names": ["hf_text"],
                "model.hf_text.checkpoint_name": args.backbone,
            },
            num_classes=num_classes,
        )
    elif args.type == "image":
        predictor = MultiModalPredictor(
            problem_type="feature_extraction",
            hyperparameters={
                "model.names": ["timm_image"],
                "model.timm_image.checkpoint_name": args.backbone,
            },
            num_classes=num_classes,
        )
    else:
        predictor = MultiModalPredictor(
            problem_type="zero_shot_image_classification",
            hyperparameters={
                "model.names": ["clip"],
                "model.clip.checkpoint_name": args.backbone,
            },
        )

    if args.type == "clip":
        clip_weights = generate_clip_weights(
            args=args,
            classnames=dataset.classnames,
            template=dataset.template,
            predictor=predictor,
        )

    bank_keys, bank_labels = generate_bank_model(
        args=args,
        train_df=train_df,
        predictor=predictor,
    )

    val_features, val_labels, test_features, test_labels = extract_val_test(
        args=args,
        predictor=predictor,
        val_df=val_df,
        test_df=test_df,
    )

    loader = build_data_loader(
        data_source=train_df,
        batch_size=256,
        shuffle=True,
        column_names=args.column_names,
        label_column=args.label_column,
    )

    memory_bank_model = AutoMMMemoryBank(
        bank_keys=bank_keys,
        bank_labels=bank_labels,
        hidden_size=test_features.shape[1],
        num_classes=num_classes,
        clip_weights=clip_weights if args.type == "clip" else None,
        model_head_type=args.model_head_type,
    ).cuda()

    run_memory_bank(
        args=args,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_features,
        test_labels=test_labels,
        predictor=predictor,
        memory_bank_model=memory_bank_model,
        loader=loader,
    )


if __name__ == "__main__":
    args = get_args()
    print(args)

    main()
