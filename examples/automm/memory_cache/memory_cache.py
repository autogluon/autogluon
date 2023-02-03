"""
    The derivative application of Tip-Adapter (https://arxiv.org/pdf/2207.09519.pdf).
    Refer to https://github.com/gaopengcuhk/Tip-Adapter
"""

from autogluon.multimodal import MultiModalPredictor
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from utils import *


def get_args():
    parser = argparse.ArgumentParser(description="MultiModal Fewshot training with memory cache.")
    parser.add_argument(
        "--type",
        type=str,
        default="clip", 
        help="Choose type of few-shot learning from 'clip', 'text', 'image' for different backbone methods.",
    )
    parser.add_argument("--backbone", type=str, default=None, help="The backbone model of MultiModal Predictor.")
    parser.add_argument("--data_path", type=str, default="./data/", help="The path for image dataset.")
    parser.add_argument("--dataset", type=str, default="food101", help="The name of dataset. Support image datasets in COOP and text datasets in Huggingface.")
    parser.add_argument("--column_names", nargs="+", default=None, help="The name of the data column.")
    parser.add_argument("--label_column", type=str, default="label", help="The name of the label column.")
    parser.add_argument("--shots", type=int, default=16, help="The shots for each class in training set.")
    parser.add_argument("--aug_epochs", type=int, default=1, help="The epochs to create the cache.")
    parser.add_argument("--lr", type=float, default=1e-2, help="The learning rate for training the model head.")
    parser.add_argument("--lr_F", type=float, default=1e-3, help="The learning rate for finetuing the memory cache.")
    parser.add_argument("--train_epoch", type=int, default=20, help="The training epochs for training the model head.")
    parser.add_argument("--train_epoch_F", type=int, default=20, help="The training epochs for finetuning the memory cache.")
    parser.add_argument("--init_alpha", type=float, default=1.17, help="The initial value of hyper-parameter alpha in memory cache. Alpha adjusts the weight of probability between the classifier and memory cache.")
    parser.add_argument("--init_beta", type=float, default=1., help="The initial values of hyper-parameter beta in memory cache. Beta modulates the sharpness when converting the similarities into non-negative values.")
    parser.add_argument("--search_scale", type=list, default=[10, 10], help="The search scale of alpha and beta.")
    parser.add_argument("--search_step", type=list, default=[200, 20], help="The steps of searching hyper-parameters alpha and beta.")
    args = parser.parse_args()

    args.cache_dir = os.path.join('./caches', args.dataset)
    os.makedirs(args.cache_dir, exist_ok=True)

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


class AutoMMMemoryCache(nn.Module):
    """
    The model to generate few shot predict probability with 
    features extracted by AutoGluon MultiModal Predictor.
    """

    def __init__(
        self, 
        cache_keys, 
        cache_values,
        hidden_size, 
        num_classes,
        clip_weights=None, 
    ):
        """
        Create the model head and the memory cache.

        Parameters
        ----------
        cache_keys
            The content of cache composed of features in the training set.
        cache_values 
            The one-hot encoded label of corresponding cache_keys.
        hidden_size
            The size of features.
        num_classes
            The classes of the dataset.
        clip_weights
            The encoded label-text prompts which is only used in type "clip".
        """
        super(AutoMMCacheAdapter, self).__init__()
        self.cache_keys = cache_keys
        self.cache_values = cache_values
        self.adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False)
        self.adapter.weight = nn.Parameter(cache_keys.t())

        self.clip_weights = clip_weights

        self.model_head = nn.Linear(hidden_size, num_classes, bias=True) if clip_weights is None else None

    def get_adapter_logits(self, affinity, pure_logits, alpha, beta):
        """
        Generate logits with memory cache based on pure_logits and cache output.

        Parameters
        ----------
        affinity
            The result of cache similarity. It is based on cosine similarity or a projector initialized with cache_keys.
        pure_logits
            The predict probability of the classifier.
        alpha
            The hyper-parameters of cache model.
        beta
            The hyper-parameters of cache model.
        
        Return
        ------
            The logits with memory cache.
        """
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.cache_values
        logits = pure_logits + cache_logits * alpha
        return logits
    
    def change_head_state(self, grad_state):
        """
        Change the training state of model head.

        Parameters
        ----------
        grad_state
            The needed training state of the model head.
        """
        if self.model_head is not None:
            for param in self.model_head.parameters():
                param.requires_grad = grad_state
    
    def change_adapter_state(self, grad_state):
        """
        Change the training state of memory cache.

        Parameters
        ----------
        grad_state
            The needed training state of the memory cache.
        """
        for param in self.adapter.parameters():
            param.requires_grad = grad_state

    def forward(self, x, alpha = 1, beta = 1):
        """
        Generate three types of logits with features.

        Parameters
        ----------
        x
            The image/text features generated by AutoGluon Multimodal Predictor.
        alpha
            The hyper-parameters of memory cache model.
        beta
            The hyper-parameters of memory cache model.
        
        Return
        ------
        The predict probability of the feature.
            - "pure_logits"
                The predict probability of classifier.
            - "logits_with_adapter"
                The predict probability composed of classifier and memory cache similarity result.
            - "logits_with_finetuned_adapter"
                The predict probability composed of classifier and fine-tuned memory cache result.
        """
        if self.clip_weights is not None:
            pure_logits = 100. * x @ self.clip_weights
        else:
            pure_logits = self.model_head(x)
        
        affinity = x @ self.cache_keys
        logits_with_adapter = self.get_adapter_logits(affinity, pure_logits, alpha, beta)

        finetuned_affinity = self.adapter(x)
        logits_with_finetuned_adapter = self.get_adapter_logits(finetuned_affinity, pure_logits, alpha, beta)

        return {
            "pure_logits": pure_logits,
            "logits_with_adapter": logits_with_adapter,
            "logits_with_finetuned_adapter": logits_with_finetuned_adapter,
        }


def train_logits(
    args, 
    val_features, 
    val_labels, 
    test_features, 
    test_labels, 
    predictor, 
    memory_cache_model, 
    alpha, 
    beta, 
    loader, 
    logits_type="pure_logits"
):
    """
    The training process of AutoMMMemoryCache.

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
    memory_cache_model
        The AutoMMCacheAdapter to generate logits and logits with memory cache.
    alpha, beta
        The hyper-parameters of memory cache.
    loader
        The dataset loader for training.
    logits_type
        The target logits of training corresponding to "pure_logits", "logits_with_adapter", "logits_with_finetuned_adapter".
    
    Return
    ------
    The best model after training.
    """
    if logits_type != "logits_with_finetuned_adapter":
        memory_cache_model.change_head_state(grad_state=True)
        memory_cache_model.change_adapter_state(grad_state=False)
        lr, train_epoch = args.lr, args.train_epoch
    else:
        memory_cache_model.change_head_state(grad_state=False)
        memory_cache_model.change_adapter_state(grad_state=True)
        lr, train_epoch = args.lr_F, args.train_epoch_F
            
    optimizer = torch.optim.AdamW(memory_cache_model.parameters(), lr=lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epoch * len(loader))
    memory_cache_model.train()

    best_acc, best_epoch = 0.0, 0
    for train_idx in range(train_epoch):
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, train_epoch))

        for i, (data, target) in enumerate(tqdm(loader)):
            target = target.cuda()
            with torch.no_grad():
                features = extract_embedding(args, data, predictor, args.column_names)
                features /= features.norm(dim=-1, keepdim=True)
            
            logits = memory_cache_model(features, alpha, beta)

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
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        memory_cache_model.eval()
        logits = memory_cache_model(val_features, alpha, beta)[logits_type]
        acc = cls_acc(logits, val_labels)
        print("**** val accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(memory_cache_model.state_dict(), args.cache_dir + "/best_F_" + str(args.shots) + "shots.pt")
    
    memory_cache_model.load_state_dict(torch.load(args.cache_dir + "/best_F_" + str(args.shots) + "shots.pt"))
    print(f"**** After fine-tuning {logits_type}, best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")
    
    return memory_cache_model


def run_memory_cache(
    args, 
    val_features, 
    val_labels, 
    test_features, 
    test_labels, 
    predictor, 
    memory_cache_model, 
    loader,
):
    """
    Test the effectiveness of memory cache. 
    Compare the results of pure_logits, logits_with_adapter and logits_with_finetuned_adapter.

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
    memory_cache_model
        The AutoMMCacheAdapter to generate logits and logits with memory cache.
    loader
        The dataset loader for training.
    """
    beta, alpha = args.init_beta, args.init_alpha

    if args.type != "clip":
        memory_cache_model = train_logits(
            args=args,
            val_features=val_features,
            val_labels=val_labels, 
            test_features=test_features, 
            test_labels=test_labels, 
            predictor=predictor, 
            memory_cache_model=memory_cache_model, 
            alpha=alpha, 
            beta=beta, 
            loader=loader, 
            logits_type="pure_logits",
        )
    
    with torch.no_grad():
        logits = memory_cache_model(val_features, alpha, beta)
    acc = cls_acc(logits["pure_logits"], val_labels)
    print("\n**** Pure model's val accuracy: {:.2f}. ****\n".format(acc))
    
    acc = cls_acc(logits["logits_with_adapter"], val_labels)
    print("**** Model with memory_cache's val accuracy: {:.2f}. ****\n".format(acc))

    best_beta, best_alpha = search_hp(
        args=args, 
        features=val_features, 
        labels=val_labels, 
        memory_cache_model=memory_cache_model, 
        logits_type="logits_with_adapter",
    )

    print("\n-------- Evaluating on the test set. --------")
    
    with torch.no_grad():
        logits = memory_cache_model(test_features, best_alpha, best_beta)
    acc = cls_acc(logits["pure_logits"], test_labels)
    print("\n**** Pure Model's test accuracy: {:.2f}. ****\n".format(acc))
 
    acc = cls_acc(logits["logits_with_adapter"], test_labels)
    print("**** Model with Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    print("\n-------- Finetune Adapter. --------")
    memory_cache_model = train_logits(
        args=args, 
        val_features=val_features, 
        val_labels=val_labels, 
        test_features=test_features, 
        test_labels=test_labels, 
        predictor=predictor, 
        memory_cache_model=memory_cache_model, 
        alpha=alpha, 
        beta=beta, 
        loader=loader, 
        logits_type="logits_with_finetuned_adapter",
    )
    best_beta, best_alpha = search_hp(
        args=args, 
        features=val_features, 
        labels=val_labels, 
        memory_cache_model=memory_cache_model, 
        logits_type="logits_with_finetuned_adapter",
    )
    with torch.no_grad():
        logits = memory_cache_model(test_features, best_alpha, best_beta)
    acc = cls_acc(logits["logits_with_finetuned_adapter"], test_labels)
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
            num_classes = num_classes,
        )
    elif args.type == "image":
        predictor = MultiModalPredictor(
            problem_type="feature_extraction",
            hyperparameters={
                "model.names": ["timm_image"],
                "model.timm_image.checkpoint_name": args.backbone,
            },
            num_classes = num_classes,
        )
    else:
        predictor = MultiModalPredictor(
            problem_type="zero_shot_image_classification",
            hyperparameters={
                "model.names": ["clip"],
                "model.clip.checkpoint_name": args.backbone,
            }
        )

    if args.type == "clip":
        clip_weights = generate_clip_weights(
            args=args,
            classnames=dataset.classnames, 
            template=dataset.template, 
            predictor=predictor,
        )

    cache_keys, cache_values = generate_cache_model(
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
    
    memory_cache_model = AutoMMMemoryCache(
        cache_keys=cache_keys, 
        cache_values=cache_values, 
        hidden_size=test_features.shape[1], 
        num_classes=num_classes, 
        clip_weights=clip_weights if args.type == "clip" else None, 
    ).cuda()

    run_memory_cache(
        args=args, 
        val_features=val_features, 
        val_labels=val_labels, 
        test_features=test_features, 
        test_labels=test_labels, 
        predictor=predictor, 
        memory_cache_model=memory_cache_model, 
        loader=loader,
    )


if __name__ == "__main__":
    args = get_args()
    print(args)

    main()
