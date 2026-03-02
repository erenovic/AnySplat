from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerContiguousCfg:
    name: Literal["contiguous"]
    num_views: int  # number of consecutive frames to return


class ViewSamplerContiguous(ViewSampler[ViewSamplerContiguousCfg]):
    """Return ``num_views`` temporally consecutive frames.

    A random start index is drawn uniformly from ``[0, N - num_views]``.
    For test / overfit stages the start is fixed at 0.

    This is useful for video-generative models (e.g. WAN / Krea) where the
    VAE requires temporally coherent clips and the frame ordering must be
    preserved.
    """

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> Int64[Tensor, " view"]:
        N = extrinsics.shape[0]
        n = self.cfg.num_views

        if N < n:
            raise ValueError(f"Scene '{scene}' has only {N} frames but {n} requested.")

        if self.stage == "test" or self.is_overfitting:
            start = 0
        else:
            start = int(torch.randint(0, N - n + 1, ()).item())

        return torch.arange(start, start + n, dtype=torch.int64, device=device)

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_views

    @property
    def num_target_views(self) -> int:
        return 0
