
import einops
import einx
import torch
import torch.nn as nn


class Tab2DEmbeddingX(torch.nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.dim = dim
        self.x_embedding = nn.Linear(1, dim)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = einx.rearrange('b s f -> b s f 1', x)
        x = self.x_embedding(x)

        return x

        

class Tab2DQuantileEmbeddingX(torch.nn.Module):

    def __init__(
            self,
            dim: int,
        ) -> None:
        
        super().__init__()

        self.dim = dim

        
    def forward(
        self, 
        x_support: torch.Tensor, 
        x_query__: torch.Tensor, 
        padding_mask: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        """
        Syntax:
        b = batch size
        s = number of observations
        f = number of features
        q = number of quantiles
        """

        batch_size = padding_mask.shape[0]
        seq_len = einx.sum('b [s]', ~padding_mask)
        feature_count = einx.sum('b [f]', ~feature_mask)

        # By setting the padded tokens to 9999 we ensure they don't participate in the quantile calculation
        x_support[padding_mask] = 9999

        q = torch.arange(1, 1000, dtype=torch.float, device=x_support.device) / 1000
        quantiles = torch.quantile(x_support, q=q, dim=1)
        quantiles = einx.rearrange('q b f -> (b f) q', quantiles)
        x_support = einx.rearrange('b s f -> (b f) s', x_support).contiguous()
        x_query__ = einx.rearrange('b s f -> (b f) s', x_query__).contiguous()

        bucketize = torch.vmap(torch.bucketize, in_dims=(0, 0), out_dims=0)
        x_support = bucketize(x_support, quantiles).float() 
        x_query__ = bucketize(x_query__, quantiles).float()
        x_support = einx.rearrange('(b f) s -> b s f', x_support, b=batch_size).contiguous()
        x_query__ = einx.rearrange('(b f) s -> b s f', x_query__, b=batch_size).contiguous()

        # If 30% is padded, the minimum will have quantile 0.0 and the maximum will have quantile 0.7 times max_length.
        # Here we correct the quantiles so that the minimum has quantile 0.0 and the maximum has quantile 1.0.
        x_support = x_support / seq_len[:, None, None] 
        x_query__ = x_query__ / seq_len[:, None, None]

        # Make sure that the padding is not used in the calculation of the mean
        x_support[padding_mask] = 0
        x_support_mean = einx.sum('b [s] f', x_support, keepdims=True) / seq_len[:, None, None]

        x_support = x_support - x_support_mean
        x_query__ = x_query__ - x_support_mean

        # Make sure that the padding is not used in the calculation of the variance
        x_support[padding_mask] = 0
        x_support_var = einx.sum('b [s] f', x_support**2, keepdims=True) / seq_len[:, None, None]

        x_support = x_support / x_support_var.sqrt()
        x_query__ = x_query__ / x_support_var.sqrt()

        # In case an x_support feature column contains one unique feature, set the feature to zero 
        x_support = torch.where(x_support_var == 0, 0, x_support)
        x_query__ = torch.where(x_support_var == 0, 0, x_query__)

        return x_support, x_query__


class Tab2DEmbeddingY(torch.nn.Module):

    def __init__(self, dim: int, n_classes: int) -> None:
        super().__init__()

        self.dim = dim
        self.n_classes = n_classes
        self.y_embedding_support = nn.Linear(1, dim)
        self.y_embedding_query = nn.Embedding(1, dim)


    def forward(self, y_support: torch.Tensor, padding_obs_support: torch.Tensor, n_obs_query: int) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size = y_support.shape[0]

        y_support = y_support.type(torch.float32)
        y_support = y_support / self.n_classes - 0.5
        y_support = einops.rearrange(y_support, 'b n -> b n 1')

        y_support = self.y_embedding_support(y_support)
        y_support[padding_obs_support] = 0

        y_query = torch.zeros((batch_size, n_obs_query, 1), device=y_support.device, dtype=torch.int64)
        y_query = self.y_embedding_query(y_query)

        return y_support, y_query
    

class Tab2DEmbeddingYClasses(torch.nn.Module):

    def __init__(
            self,
            dim: int,
            n_classes: int, 
        ) -> None:
        
        super().__init__()

        self.n_classes = n_classes
        self.dim = dim

        self.y_embedding = nn.Embedding(n_classes, dim,)
        self.y_mask = nn.Embedding(1, dim) # masking is also modeled as a separate class


    def forward(self, y_support: torch.Tensor, padding_obs_support: torch.Tensor, n_obs_query: int) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size = y_support.shape[0]
        n_obs_support = y_support.shape[1]

        y_support = y_support.type(torch.int64)
        y_support = einops.rearrange(y_support, 'b n -> b n 1')
        y_support[padding_obs_support] = 0    # padded tokens are -100 -> set it to zero so nn.Embedding can handle it
        y_support = self.y_embedding(y_support)
        y_support[padding_obs_support] = 0    # just to make sure, set it to zero again
         
        y_query = torch.zeros((batch_size, n_obs_query, 1), device=y_support.device, dtype=torch.int64)
        y_query = self.y_mask(y_query)

        return y_support, y_query
    

class Tab2DEmbeddingYRegression(torch.nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.dim = dim
        self.y_embedding = nn.Linear(1, dim)
        self.y_mask = nn.Embedding(1, dim)


    def forward(self, y_support: torch.Tensor, padding_obs_support: torch.Tensor, n_obs_query: int) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size = y_support.shape[0]
        y_support = y_support.type(torch.float32)
        y_support = einops.rearrange(y_support, 'b n -> b n 1')
        y_support = self.y_embedding(y_support)
        y_support[padding_obs_support] = 0

        y_query = torch.zeros((batch_size, n_obs_query, 1), device=y_support.device, dtype=torch.int64)
        y_query = self.y_mask(y_query)

        return y_support, y_query