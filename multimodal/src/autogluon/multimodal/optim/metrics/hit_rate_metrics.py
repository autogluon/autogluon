from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchmetrics


class CustomHitRate(torchmetrics.Metric):
    """
    Compute the hit rate when doing semantic search between two group of embeddings.
    We assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.add_state("query_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("response_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("logit_scale", default=[], dist_reduce_fx=None)

    def update(
        self,
        batch_query_embeds: torch.Tensor,
        batch_response_embeds: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
    ):
        self.query_embeddings.append(batch_query_embeds)
        self.response_embeddings.append(batch_response_embeds)
        if logit_scale is not None:
            self.logit_scale.append(logit_scale)

    def compute(self):
        query_embeddings = torch.cat(self.query_embeddings)
        response_embeddings = torch.cat(self.response_embeddings)
        if self.logit_scale:
            logit_scale = torch.mean(torch.stack(self.logit_scale))
        else:
            logit_scale = 1

        return self.compute_hit_rate(query_embeddings, response_embeddings, logit_scale)

    @staticmethod
    def compute_hit_rate(features_a, features_b, logit_scale, top_ks=[1, 5, 10]):
        """
        Compute symmetric hit rates between two groups of features.

        Parameters
        ----------
        features_a
            One group of features.
        features_b
            The other group of features.
        logit_scale
            The scale of logit (Used in CLIP).
        top_ks
            Consider only the top k elements for each query.

        Returns
        -------
        The accumulated hit rate.
        """
        assert len(features_a) == len(features_b)
        hit_rate = 0
        logits_per_a = (logit_scale * features_a @ features_b.t()).detach().cpu()
        logits_per_b = logits_per_a.t().detach().cpu()

        logits = {"logits_per_a": logits_per_a, "logits_per_b": logits_per_b}
        ground_truth = torch.arange(len(features_b)).view(-1, 1)

        for name, logit in logits.items():
            ranking = torch.argsort(logit, descending=True)
            preds = torch.where(ranking == ground_truth)[1]

            for k in top_ks:
                hit_rate += (preds < k).float().mean()

        hit_rate /= len(top_ks) * len(logits)
        return hit_rate
