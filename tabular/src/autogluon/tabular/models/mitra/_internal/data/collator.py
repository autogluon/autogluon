import torch


class CollatorWithPadding():

    def __init__(
            self, 
            max_features: int,
            pad_to_max_features: bool,
        ) -> None:
        
        self.max_features = max_features
        self.pad_to_max_features = pad_to_max_features


    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:

        max_support_samples = max(dataset['x_support'].shape[0] for dataset in batch)
        max_query_samples = max(dataset['x_query'].shape[0] for dataset in batch)
        max_features = max(dataset['x_support'].shape[1] for dataset in batch)

        if self.pad_to_max_features:
            max_features = self.max_features

        batch_size = len(batch)

        tensor_dict = {
            'x_support': torch.zeros((batch_size, max_support_samples, max_features), dtype=batch[0]['x_support'].dtype),
            'y_support': torch.full((batch_size, max_support_samples), fill_value=-100, dtype=batch[0]['y_support'].dtype),
            'x_query': torch.zeros((batch_size, max_query_samples, max_features), dtype=batch[0]['x_query'].dtype),
            'y_query': torch.full((batch_size, max_query_samples), fill_value=-100, dtype=batch[0]['y_query'].dtype),
            'padding_features': torch.ones((batch_size, max_features), dtype=torch.bool),
            'padding_obs_support': torch.ones((batch_size, max_support_samples), dtype=torch.bool),
            'padding_obs_query': torch.ones((batch_size, max_query_samples), dtype=torch.bool),
        }

        for i, dataset in enumerate(batch):
            tensor_dict['x_support'][i, :dataset['x_support'].shape[0], :dataset['x_support'].shape[1]] = dataset['x_support']
            tensor_dict['y_support'][i, :dataset['y_support'].shape[0]] = dataset['y_support']
            tensor_dict['x_query'][i, :dataset['x_query'].shape[0], :dataset['x_support'].shape[1]] = dataset['x_query']
            tensor_dict['y_query'][i, :dataset['y_query'].shape[0]] = dataset['y_query']
            tensor_dict['padding_features'][i, :dataset['x_support'].shape[1]] = False
            tensor_dict['padding_obs_support'][i, :dataset['x_support'].shape[0]] = False
            tensor_dict['padding_obs_query'][i, :dataset['x_query'].shape[0]] = False

        return tensor_dict
