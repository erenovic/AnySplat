from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .adaptive_sampling import pose_distance
from .view_sampler import ViewSampler


@dataclass
class ViewSamplerSequentialCfg:
    name: Literal["sequential"]
    num_views_per_scene: int  # views returned per scene group
    num_scenes: int  # number of scene groups per sample
    # When True, the last frame of each scene is duplicated as the first frame
    # of the next, so consecutive scenes share one boundary view.
    overlap_one_frame: bool = False
    # Scale applied to R_measure when computing the SE(3) FPS distance.
    # 1.0 -> rotation and normalised translation contribute equally.
    # >1  -> orientation diversity emphasised; <1 -> position diversity emphasised.
    rotation_scale: float = 2.0
    # Fraction of the video (by proximity to the random start frame) that forms
    # the FPS candidate set.  Directly controls locality of the selected views.
    # 0.0 -> (clamped to n frames) only the n closest frames to start — maximum
    #        overlap, minimum diversity.
    # 1.0 -> all N frames — maximum diversity, same as running FPS over the
    #        whole video.
    # 0.2 -> the 20% of frames closest to start — local cluster, balanced.
    neighbour_scale: float = 0.3

    def __post_init__(self):
        # Unique frames FPS must select (shared boundary frames are not duplicated yet).
        self.num_views = self.num_views_per_scene * self.num_scenes - (self.num_scenes - 1) * int(
            self.overlap_one_frame
        )
        # Group size used by the expansion step in sample().
        self.group_size = self.num_views_per_scene if self.overlap_one_frame else 0


class ViewSamplerSequential(ViewSampler[ViewSamplerSequentialCfg]):
    """Sample num_views spatially spread frames from a proximity-ranked window.

    A random start frame is drawn, then the neighbour_scale * N frames closest
    to it (by SE(3) distance) form the candidate pool.  FPS on those candidates
    selects the final num_views frames, balancing diversity and overlap.

    neighbour_scale directly controls locality: 1.0 = entire video, 0.2 = local
    20%-closest cluster.
    """

    @staticmethod
    def _fps_pose_distance(dist_matrix: Tensor, n: int, start: int = -1) -> Tensor:
        """Farthest-point sampling given a precomputed (N, N) distance matrix.

        If start >= 0, seeds from that index; otherwise seeds from the most
        peripheral frame (highest mean distance from all others).
        """
        device = dist_matrix.device

        selected = torch.zeros(n, dtype=torch.long, device=device)
        selected[0] = start if start >= 0 else dist_matrix.mean(0).argmax()

        min_dists = dist_matrix[selected[0]].clone()
        for i in range(1, n):
            selected[i] = min_dists.argmax()
            min_dists = torch.minimum(min_dists, dist_matrix[selected[i]])

        return selected

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> Int64[Tensor, " view"]:
        N = extrinsics.shape[0]
        n = self.cfg.num_views

        if N < n:
            raise ValueError(f"Scene '{scene}' has only {N} frames but {n} requested.")

        # Precompute the full N x N normalised SE(3) distance matrix once.
        _, R_all, t_all = pose_distance(extrinsics, extrinsics)  # (N, N) each
        cam_pos = extrinsics[:, :3, 3]
        scene_scale = (cam_pos - cam_pos.mean(0)).norm(dim=-1).max().clamp(min=1e-6)
        t_norm = t_all / scene_scale
        dist_all = (t_norm**2 + (self.cfg.rotation_scale * R_all) ** 2).sqrt()  # (N, N)

        # Random start frame (deterministic for test/overfit).
        if self.stage == "test" or self.is_overfitting:
            start = int(dist_all.mean(0).argmax().item())
        else:
            start = int(torch.randint(0, N, (), device=device).item())

        # Candidate set: the (neighbour_scale * N) frames closest to `start`.
        # neighbour_scale=1.0 -> all frames; 0.2 -> the closest 20%; etc.
        # Clamped to at least n so FPS always has enough frames to choose from.
        n_cands = max(n, round(self.cfg.neighbour_scale * N))
        candidates = dist_all[start].argsort()[:n_cands]  # (C,)

        # FPS on the candidate set using the precomputed submatrix.
        dist_cand = dist_all[candidates][:, candidates]  # (C, C)
        fps_cand = self._fps_pose_distance(dist_cand, n)
        view_indices = candidates[fps_cand].sort().values  # (n,) sorted

        # Expand with scene-boundary overlap: each group's last frame becomes the
        # first frame of the next group, so consecutive scenes share one view.
        if self.cfg.group_size > 0:
            k = self.cfg.group_size
            stride = k - 1
            n_groups = (n - 1) // stride
            expand_idx = torch.cat([torch.arange(g * stride, g * stride + k) for g in range(n_groups)])
            view_indices = view_indices[expand_idx]  # (n_groups * k,)

        return view_indices.to(torch.int64)

    @property
    def num_context_views(self) -> int:
        if self.cfg.group_size > 0:
            k = self.cfg.group_size
            n_groups = (self.cfg.num_views - 1) // (k - 1)
            return k * n_groups
        return self.cfg.num_views

    @property
    def num_target_views(self) -> int:
        return 0
