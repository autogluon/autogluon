import copy
import torch


class ContrastiveTransformations:
    """
    Support within-batch tabular data corruption/augmentation.
    Randomly permutate the values across rows with a probability.
    """

    def __init__(self, model, mode, problem_type, corruption_rate):
        """
        Parameters
        ----------
        model
            The AutoMM model used in pretraining.
        mode
            Which corruption to use. Support random permutation or no corruption.
        problem_type
            The type of down-stream tasks (regression/binary/multiclass). Reserved for task-specific corruption.
        corruption_rate
            The probability of a tabular field gots corrupted, 0<=corruption_rate<=1.
        """
        self.model = model
        self.mode = mode if mode is not None else "identical"
        self.problem_type = problem_type
        self.corruption_rate = corruption_rate
        self.last_batch = None

    def __call__(self, batch):
        if self.mode == "identical":
            return self.identical(batch)
        elif self.mode == "permutation":
            return self.random_perm(batch)
        else:
            raise ValueError(
                f"Current mode {self.mode} is not supported."
                "Consider choosing from the following options:"
                "identical, permutation."
            )

    def identical(self, batch):
        batch = copy.deepcopy(batch)
        return batch

    def random_perm(self, batch):
        corruption_rate = self.corruption_rate
        (batch_size,) = batch[self.model.label_key].size()
        batch = copy.deepcopy(batch)

        num_features = 0
        for permodel in self.model.model:
            if hasattr(permodel, "categorical_key"):
                num_features += len(batch[permodel.categorical_key])
            if hasattr(permodel, "numerical_key"):
                _, m = batch[permodel.numerical_key].size()
                num_features += m

        corruption_mask = torch.zeros(
            batch_size, num_features, dtype=torch.bool, device=batch[self.model.label_key].device
        )
        corruption_len = int(num_features * corruption_rate)
        for i in range(batch_size):
            corruption_idx = torch.randperm(num_features)[:corruption_len]
            corruption_mask[i, corruption_idx] = True
        feature_idx = 0

        for permodel in self.model.model:
            if hasattr(permodel, "categorical_key"):
                categorical_features = []
                for categorical_feature in batch[permodel.categorical_key]:
                    random_idx = torch.randint(high=batch_size, size=(batch_size,))
                    random_sample = categorical_feature[random_idx].clone()
                    positive = torch.where(corruption_mask[:, feature_idx], random_sample, categorical_feature)
                    feature_idx += 1
                    categorical_features.append(positive)
                batch[permodel.categorical_key] = tuple(categorical_features)
            if hasattr(permodel, "numerical_key"):
                numerical_features = batch[permodel.numerical_key]
                _, m = numerical_features.size()
                indices = torch.randint(high=batch_size, size=(batch_size, m))
                random_sample = numerical_features[indices, torch.arange(m).unsqueeze(0)].clone()
                batch[permodel.numerical_key] = torch.where(
                    corruption_mask[:, feature_idx : feature_idx + m], random_sample, numerical_features
                )
                feature_idx += m
        return batch
