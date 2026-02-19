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
    neighbour_scale: float = 1.0
    # Minimum normalised SE(3) distance between consecutive frames in the output
    # sequence.  Expressed as a fraction of the median pairwise distance among
    # the FPS-selected frames.  0.0 disables the minimum constraint.
    min_pair_distance: float = 0.0
    # Maximum normalised SE(3) distance between consecutive frames.  Expressed
    # as a fraction of the median pairwise distance among FPS-selected frames.
    # 0.0 disables the maximum constraint.
    max_pair_distance: float = 0.0

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

    @staticmethod
    def _sequential_chain(
        dist_matrix: Tensor, candidates: Tensor, n: int, min_d: float, max_d: float, start: int = -1
    ) -> Tensor:
        """Build a monotonically-increasing chain of *n* frames from *candidates*.

        Sorts *candidates* by video index, then walks forward: from the current
        frame, picks the next frame (by video index) whose SE(3) distance lies
        in [min_d, max_d].  This guarantees the output indices are strictly
        increasing while enforcing spacing.

        The candidate pool should be larger than n so the chain can skip frames
        that are too close without running out.

        When no future frame satisfies both bounds, constraints are relaxed
        progressively: first drop the max bound, then drop both bounds.  Each
        candidate is used at most once so indices are never repeated.

        Args:
            dist_matrix: (N, N) pairwise distance matrix (full, not subsetted).
            candidates:  (C,) indices into dist_matrix (C >= n).
            n:           number of frames to select.
            min_d:       absolute minimum distance between consecutive frames.
            max_d:       absolute maximum distance (0 = disabled).

        Returns:
            (n,) indices in strictly increasing order forming the chain.
        """
        # Sort candidates by video index so we only walk forward.
        sorted_cands = candidates.sort().values  # (C,)
        C = sorted_cands.shape[0]
        device = sorted_cands.device

        chain = torch.zeros(n, dtype=torch.long, device=device)
        # Seed from the requested start frame if it is in the candidate set,
        # otherwise fall back to the first (smallest-index) candidate.
        seed_pos = 0
        if start >= 0 and (sorted_cands == start).any():
            seed_pos = int((sorted_cands == start).nonzero(as_tuple=True)[0][0].item())
        chain[0] = sorted_cands[seed_pos]
        used = {seed_pos}  # indices into sorted_cands that have been consumed
        ptr = seed_pos  # pointer into sorted_cands

        for i in range(1, n):
            cur = chain[i - 1].item()
            best_idx = -1

            # Pass 1: first unused candidate satisfying both [min_d, max_d].
            for j in range(ptr + 1, C):
                if j in used:
                    continue
                d = dist_matrix[cur, sorted_cands[j].item()].item()
                if d >= min_d and (max_d <= 0 or d <= max_d):
                    best_idx = j
                    break

            # Pass 2: relax max — only enforce min_d.
            if best_idx < 0 and min_d > 0:
                for j in range(ptr + 1, C):
                    if j in used:
                        continue
                    d = dist_matrix[cur, sorted_cands[j].item()].item()
                    if d >= min_d:
                        best_idx = j
                        break

            # Pass 3: any unused candidate forward of ptr.
            if best_idx < 0:
                for j in range(ptr + 1, C):
                    if j not in used:
                        best_idx = j
                        break

            # Pass 4: any unused candidate at all (avoids repeating indices).
            if best_idx < 0:
                for j in range(C):
                    if j not in used:
                        best_idx = j
                        break

            # Truly exhausted (C < n) — duplicate last as last resort.
            if best_idx < 0:
                best_idx = ptr

            chain[i] = sorted_cands[best_idx]
            used.add(best_idx)
            ptr = best_idx

        return chain

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
        # Cap so that at least 2*n frames remain after `start`, giving the
        # sequential chain enough slack to skip close frames for spacing.
        max_start = max(0, N - 2 * n)
        if self.stage == "test" or self.is_overfitting:
            start = int(dist_all[:max_start + 1].mean(0).argmax().item()) if max_start > 0 else 0
        else:
            start = int(torch.randint(0, max_start + 1, (), device=device).item())

        # Candidate set: the (neighbour_scale * N) frames closest to `start`.
        # neighbour_scale=1.0 -> all frames; 0.2 -> the closest 20%; etc.
        # Clamped to at least n so FPS always has enough frames to choose from.
        n_cands = max(n, round(self.cfg.neighbour_scale * N))
        candidates = dist_all[start].argsort()[:n_cands]  # (C,)

        # FPS on the candidate set using the precomputed submatrix.
        dist_cand = dist_all[candidates][:, candidates]  # (C, C)
        fps_cand = self._fps_pose_distance(dist_cand, n)
        fps_global = candidates[fps_cand]  # (n,) indices into dist_all

        # Order the FPS-selected frames into a chain with spacing constraints.
        if self.cfg.min_pair_distance > 0 or self.cfg.max_pair_distance > 0:
            # Compute absolute thresholds from the median pairwise distance
            # among the FPS-selected frames.
            fps_dist = dist_all[fps_global][:, fps_global]  # (n, n)
            triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
            median_d = fps_dist[triu_mask].median().item()
            min_d = self.cfg.min_pair_distance * median_d
            max_d = self.cfg.max_pair_distance * median_d
            # Only keep candidates at or after `start` so the chain never
            # wraps around to earlier frames.
            forward_cands = candidates[candidates >= start]
            if forward_cands.shape[0] < n:
                # Not enough forward frames — shift start earlier to guarantee n.
                forward_cands = candidates.sort().values[-n:]
                start = forward_cands[0].item()
            view_indices = self._sequential_chain(dist_all, forward_cands, n, min_d, max_d, start=start)
        else:
            view_indices = fps_global.sort().values  # (n,) original behaviour

        # Fallback: if the chain is not monotonically ascending, sample a
        # contiguous range with a random stride of 2–5.
        if (view_indices[1:] <= view_indices[:-1]).any():
            stride_fb = int(torch.randint(2, 6, (), device=device).item())
            span = stride_fb * (n - 1)  # total frames spanned
            max_fb_start = max(0, N - span - 1)
            fb_start = int(torch.randint(0, max_fb_start + 1, (), device=device).item())
            view_indices = torch.arange(fb_start, fb_start + span + 1, stride_fb, device=device)[:n]

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
