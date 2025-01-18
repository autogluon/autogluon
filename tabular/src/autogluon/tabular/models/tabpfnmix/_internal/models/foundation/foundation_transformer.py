
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from ...core.enums import Task
from .embedding import FoundationEmbeddingX, FoundationEmbeddingYFloat, FoundationEmbeddingYInteger


class FoundationTransformer(nn.Module, PyTorchModelHubMixin):

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        attn_dropout: float,
        y_as_float_embedding: bool,
        task: str = Task.CLASSIFICATION,
    ) -> None:
        
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.attn_dropout = attn_dropout
        self.y_as_float_embedding = y_as_float_embedding
        self.task = task

        self.x_embedding = FoundationEmbeddingX(dim, n_features)

        if self.y_as_float_embedding:
            self.y_embedding = FoundationEmbeddingYFloat(dim)
        else:
            self.y_embedding = FoundationEmbeddingYInteger(n_classes, dim)

        self.layers = nn.ModuleList([])

        for _ in range(n_layers):

            att = MultiheadAttention(dim, n_heads)

            self.layers.append(nn.ModuleDict({
                'layer_norm1': nn.LayerNorm(dim),
                'attention': att,
                'layer_norm2': nn.LayerNorm(dim),
                'linear1': nn.Linear(dim, dim*4),
                'linear2': nn.Linear(dim*4, dim),
            }))

        self.final_layer1 = nn.Linear(dim, dim*4)
        if self.task == Task.CLASSIFICATION:
            self.final_layer2 = nn.Linear(dim*4, n_classes)
        elif self.task == Task.REGRESSION:
            self.final_layer2 = nn.Linear(dim*4, 1)
        self.init_weights()


    def init_weights(self):

        for module_dict in self.layers:

            # module_dict['attention'].init_weights()
            nn.init.zeros_(module_dict['linear2'].weight)
            nn.init.zeros_(module_dict['linear2'].bias)
            

    def forward(self, x_support: torch.Tensor, y_support: torch.Tensor, x_query: torch.Tensor):

        """
        x_support is (batch_size, n_observations_support, n_features)
        y_support is (batch_size, n_observations_support)

        x_query is (batch_size, n_observations_query, n_features)

        returns:

        y_query is (batch_size, n_observations_query, n_classes)

        syntax:
        b = batch size
        n = number of observations
        d = dimension of embedding
        c = number of classes
        """

        x_query__ = x_query

        batch_size = x_support.shape[0]
        n_obs_support = x_support.shape[1]
        n_obs_query__ = x_query__.shape[1]

        padding_mask = torch.zeros((batch_size, n_obs_support), dtype=torch.bool, device=x_support.device)
        padding_mask[y_support == -100] = True

        x_support, x_query__ = self.x_embedding(x_support, x_query__)
        y_support, y_query__ = self.y_embedding(y_support, n_obs_query__)

        support = x_support + y_support
        query__ = x_query__ + y_query__

        x, pack = einops.pack((support, query__), 'b * d')
        
        for module_dict in self.layers:

            x_residual = x
            support, query__ = einops.unpack(x, pack, 'b * d')
            att_support = module_dict['attention'](support, support, support, key_padding_mask=padding_mask)
            att_query__ = module_dict['attention'](query__, support, support, key_padding_mask=padding_mask)
            x = einops.pack((att_support, att_query__), 'b * d')[0]
            x = x_residual + x
            x = module_dict['layer_norm1'](x)
            x_residual = x
            x = module_dict['linear1'](x)
            x = torch.nn.functional.gelu(x)
            x = module_dict['linear2'](x)
            x = x_residual + x
            x = module_dict['layer_norm2'](x)

        x = self.final_layer1(x)
        x = F.gelu(x)
        x = self.final_layer2(x)

        support, query__ = einops.unpack(x, pack, 'b * c')

        return query__



class MultiheadAttention(torch.nn.Module):

    def __init__(self, dim: int, n_heads: int) -> None:
        
        super().__init__()

        self.use_flash_attention = False
        self.dim = dim
        self.n_heads = n_heads

        self.att = nn.MultiheadAttention(dim, n_heads, dropout=0.0, batch_first=True)



    def init_weights(self):
        pass
        # nn.init.zeros_(self.att.out_proj.weight)
        # nn.init.zeros_(self.att.out_proj.bias)

    
    def forward(
            self, 
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor, 
            key_padding_mask: torch.Tensor
        ) -> torch.Tensor:
        """
        b = batch size
        n = number of samples (dataset size)
        h = heads
        d = dimension of embedding

        query is (b, n, d)
        key is (b, n, d)
        value is (b, n, d)

        attention weights will be (b, h, n, n)
        output will be (b, n, d)
        """

        output = self.att(query, key, value, key_padding_mask=key_padding_mask)[0]
        return output




class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x