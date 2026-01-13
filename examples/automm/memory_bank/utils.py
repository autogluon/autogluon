import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import list_datasets, load_dataset
from imagedatasets import build_dataset
from setfit import sample_dataset
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from autogluon.multimodal import MultiModalPredictor


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def generate_image_df(args, dataset):
    column_names = args.column_names + [args.label_column]
    dataset_df = pd.DataFrame(columns=column_names)
    for data in dataset:
        per_df = pd.DataFrame([[data.impath, data.label]], columns=column_names)
        dataset_df = pd.concat(
            [dataset_df, per_df],
            ignore_index=True,
        )
    return dataset_df


def generate_dataset(args):
    if args.type == "text":
        dataset = load_dataset(args.dataset)
        train_dataset = sample_dataset(dataset["train"], label_column=args.label_column, num_samples=args.shots)
        val_dataset = dataset["validation"] if "validation" in dataset else dataset["test"]
        test_dataset = dataset["test"]

        train_dataset.set_format("pandas")
        val_dataset.set_format("pandas")
        test_dataset.set_format("pandas")

        train_df = train_dataset[:].drop(columns=["label_text"])
        val_df = val_dataset[:].drop(columns=["label_text"])
        test_df = test_dataset[:].drop(columns=["label_text"])

        num_classes = train_dataset[args.label_column].max() + 1
    else:
        dataset = build_dataset(args.dataset, args.data_path, args.shots)
        train_df = generate_image_df(args, dataset.train_x)
        val_df = generate_image_df(args, dataset.val)
        test_df = generate_image_df(args, dataset.test)
        num_classes = len(dataset.classnames)

    return dataset, train_df, val_df, test_df, num_classes


def extract_embedding(args, data, predictor, column_names):
    features = []
    for per_name in column_names:
        per_features = torch.tensor(predictor.extract_embedding({per_name: data[per_name]})[per_name]).cuda()
        features.append(per_features)
    features = torch.stack(features, dim=1).cuda()
    features = features.mean(dim=1)
    features /= features.norm(dim=-1, keepdim=True)
    return features


def generate_clip_weights(args, classnames, template, predictor):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            classname = classname.replace("_", " ")
            text = {"text": [t.format(classname) for t in template]}
            class_embeddings = extract_embedding(args, text, predictor, ["text"])
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).cuda()

    return clip_weights


def generate_bank_model(args, train_df, predictor):
    bank_keys = []
    bank_labels = []
    with torch.no_grad():
        for augment_idx in range(args.aug_epochs):
            print("Augment Epoch: {:} / {:}".format(augment_idx, args.aug_epochs))
            train_features = extract_embedding(args, train_df, predictor, args.column_names)
            bank_keys.append(train_features.unsqueeze(0))

        for index, per_data in train_df.iterrows():
            bank_labels.append(per_data[args.label_column])

        bank_keys = torch.cat(bank_keys, dim=0).mean(dim=0)
        bank_keys /= bank_keys.norm(dim=-1, keepdim=True)
        bank_keys = bank_keys.permute(1, 0)
        bank_labels = torch.tensor(bank_labels).cuda()

    return bank_keys, bank_labels


def extract_val_test(args, predictor, val_df, test_df):
    val_features = extract_embedding(args, val_df, predictor, args.column_names)
    test_features = extract_embedding(args, test_df, predictor, args.column_names)
    val_labels = torch.tensor(val_df[args.label_column]).cuda()
    test_labels = torch.tensor(test_df[args.label_column]).cuda()
    return val_features, val_labels, test_features, test_labels


def search_hp(
    args,
    features,
    labels,
    memory_bank_model,
    logits_type=None,
):
    """
    Search the best hyper-parameters of alpha and beta.

    Parameters
    ----------
    args
        The args of searching scales and steps.
    features, labels
        The preprocessed features and labels of validation set.
    memory_bank_model
        The AutoMMMemoryBank to generate logits and logits with memory bank.
    logits_type
        The target logits of searching corresponding to "adapted_logits" and "adapted_logits_with_finetuning".

    Return
    ------
    The best hyper-parameters of alpha and beta.
    """
    beta_list = [i * (args.search_scale[0] - 0.1) / args.search_step[0] + 0.1 for i in range(args.search_step[0])]
    alpha_list = [i * (args.search_scale[1] - 0.1) / args.search_step[1] + 0.1 for i in range(args.search_step[1])]

    best_acc = 0
    best_beta, best_alpha = 0, 0

    pure_logits = None
    for beta in beta_list:
        for alpha in alpha_list:
            with torch.no_grad():
                logits = memory_bank_model(features, alpha, beta, pure_logits)
            if pure_logits is None:
                pure_logits = logits["pure_logits"]
            acc = cls_acc(logits[logits_type], labels)

            if acc > best_acc:
                print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                best_acc = acc
                best_beta = beta
                best_alpha = alpha

        print("\nAfter searching, the best accuracy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


class Wrapper(TorchDataset):
    def __init__(self, data_source, column_names=["image"], label_column="label"):
        self.data_source = data_source
        self.column_names = column_names
        self.label_column = label_column

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source.loc[idx]
        data = {}
        for per_name in self.column_names:
            data.update({per_name: item[per_name]})
        return data, item[self.label_column]


def build_data_loader(
    data_source=None,
    batch_size=64,
    num_workers=8,
    shuffle=False,
    column_names=["image"],
    label_column="label",
):
    data_loader = torch.utils.data.DataLoader(
        Wrapper(data_source, column_names),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=(torch.cuda.is_available()),
    )
    assert len(data_loader) > 0

    return data_loader
