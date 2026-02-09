"""View sampler for video-style datasets where target views follow context views.

Context frames are spread out within a window using farthest point sampling
on camera positions (from extrinsics), and target frames are the sequential
frames immediately after the last context frame.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler
from .view_sampler_bounded_v2 import farthest_point_sample


@dataclass
class ViewSamplerSequentialTargetCfg:
    name: Literal["sequential_target"]
    num_context_views: int
    num_target_views: int
    # Total context window width (in frame indices, leftmost-to-rightmost).
    min_context_gap: int
    max_context_gap: int
    # Warm-up: linearly ramp context gap from initial to final values.
    warm_up_steps: int
    initial_min_context_gap: int
    initial_max_context_gap: int
    # Number of frames to skip between the last context frame and the first
    # target frame.  1 = immediately next frame (default), 5 = skip 4 frames.
    target_offset: int = 1


class ViewSamplerSequentialTarget(ViewSampler[ViewSamplerSequentialTargetCfg]):
    """Sample spatially spread context views followed by sequential targets.

    1. Pick a window of candidate frames sized ``[min_context_gap, max_context_gap]``.
    2. Within that window, select ``num_context_views`` using farthest point
       sampling on the camera translation vectors – this maximises spatial
       spread while adapting to non-uniform camera motion.
    3. Target views are the ``num_target_views`` frames directly after the
       highest-index context frame (video continuation, never between context
       frames).
    """

    def schedule(self, initial: int, final: int) -> int:
        if self.cfg.warm_up_steps <= 0:
            return final
        fraction = min(self.global_step / self.cfg.warm_up_steps, 1.0)
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> tuple[
        Int64[Tensor, " context_view"],
        Int64[Tensor, " target_view"],
    ]:
        num_views = extrinsics.shape[0]
        n_ctx = self.cfg.num_context_views
        n_tgt = self.cfg.num_target_views
        t_off = self.cfg.target_offset  # frames to skip after last context

        # --- resolve current context gap bounds (warm-up aware) ---
        if self.stage == "test":
            min_gap = self.cfg.max_context_gap
            max_gap = self.cfg.max_context_gap
        else:
            min_gap = self.schedule(
                self.cfg.initial_min_context_gap,
                self.cfg.min_context_gap,
            )
            max_gap = self.schedule(
                self.cfg.initial_max_context_gap,
                self.cfg.max_context_gap,
            )

        # Clamp: window + offset + target frames must fit inside the scene.
        max_gap = min(max_gap, num_views - n_tgt - t_off)
        min_gap = max(min(min_gap, max_gap), n_ctx - 1)

        if max_gap < min_gap:
            raise ValueError(
                f"Scene '{scene}' has only {num_views} frames, which is too "
                f"few for context_gap={min_gap} + {n_tgt} target views."
            )

        # --- pick a random context window size ---
        context_gap = torch.randint(
            min_gap, max_gap + 1, size=(), device=device
        ).item()

        # --- pick a random starting position ---
        # The window spans [start, start + context_gap] and targets span
        # [start + context_gap + 1, ... + n_tgt], so we need
        # start + context_gap + n_tgt <= num_views - 1.
        max_start = num_views - context_gap - n_tgt - t_off
        if self.stage == "test" or self.is_overfitting:
            start = 0
        else:
            start = torch.randint(
                0, max(max_start, 0) + 1, size=(), device=device
            ).item()

        # --- select context views via farthest point sampling ---
        window_indices = torch.arange(start, start + context_gap + 1)
        # Camera positions: translation column of the extrinsics ([:3, 3])
        cam_positions = extrinsics[window_indices, :3, 3].unsqueeze(0)  # (1, W, 3)
        fps_local = farthest_point_sample(cam_positions, n_ctx).squeeze(0)  # (n_ctx,)
        # Map local window indices back to global frame indices
        context_indices = window_indices[fps_local]

        # Sort so the last element is the highest frame index – target views
        # will continue from there.
        context_indices = context_indices.sort().values

        # --- target views: sequential after the last context frame ---
        last_context = context_indices[-1].item()
        target_indices = torch.arange(n_tgt, device=device) + last_context + t_off

        # Safety clamp (should not trigger with correct gap logic).
        target_indices = target_indices.clamp(max=num_views - 1)

        return context_indices.to(torch.int64), target_indices.to(torch.int64)

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views


# """View sampler for video-style datasets where target views follow context views.

# Context frames are spread out within a window using farthest point sampling
# on camera positions (from extrinsics), and target frames are the sequential
# frames immediately after the last context frame.
# """

# from dataclasses import dataclass
# from typing import Literal

# import torch
# from jaxtyping import Float, Int64
# from torch import Tensor

# from .view_sampler import ViewSampler
# from .view_sampler_bounded_v2 import farthest_point_sample


# @dataclass
# class ViewSamplerSequentialTargetCfg:
#     name: Literal["sequential_target"]
#     num_context_views: int
#     num_target_views: int
#     # Total context window width (in frame indices, leftmost-to-rightmost).
#     min_context_gap: int
#     max_context_gap: int
#     # Warm-up: linearly ramp context gap from initial to final values.
#     warm_up_steps: int
#     initial_min_context_gap: int
#     initial_max_context_gap: int
#     # Gap (in frames) between the last context frame and the first target frame.
#     # Each sample draws uniformly from [min_target_gap, max_target_gap].
#     min_target_gap: int = 1
#     max_target_gap: int = 1


# class ViewSamplerSequentialTarget(ViewSampler[ViewSamplerSequentialTargetCfg]):
#     """Sample spatially spread context views followed by sequential targets.

#     1. Pick a window of candidate frames sized ``[min_context_gap, max_context_gap]``.
#     2. Within that window, select ``num_context_views`` using farthest point
#        sampling on the camera translation vectors – this maximises spatial
#        spread while adapting to non-uniform camera motion.
#     3. Target views are the ``num_target_views`` frames directly after the
#        highest-index context frame (video continuation, never between context
#        frames).
#     """

#     def schedule(self, initial: int, final: int) -> int:
#         if self.cfg.warm_up_steps <= 0:
#             return final
#         fraction = min(self.global_step / self.cfg.warm_up_steps, 1.0)
#         return min(initial + int((final - initial) * fraction), final)

#     def sample(
#         self,
#         scene: str,
#         extrinsics: Float[Tensor, "view 4 4"],
#         intrinsics: Float[Tensor, "view 3 3"],
#         device: torch.device = torch.device("cpu"),
#         **kwargs,
#     ) -> tuple[
#         Int64[Tensor, " context_view"],
#         Int64[Tensor, " target_view"],
#     ]:
#         num_views = extrinsics.shape[0]
#         n_ctx = self.cfg.num_context_views
#         n_tgt = self.cfg.num_target_views
#         min_t_gap = self.cfg.min_target_gap
#         max_t_gap = self.cfg.max_target_gap

#         # --- resolve current context gap bounds (warm-up aware) ---
#         if self.stage == "test":
#             min_gap = self.cfg.max_context_gap
#             max_gap = self.cfg.max_context_gap
#         else:
#             min_gap = self.schedule(
#                 self.cfg.initial_min_context_gap,
#                 self.cfg.min_context_gap,
#             )
#             max_gap = self.schedule(
#                 self.cfg.initial_max_context_gap,
#                 self.cfg.max_context_gap,
#             )

#         # Clamp: context window + target window must fit inside the scene.
#         # Worst case: context_gap + max_target_gap frames needed.
#         max_gap = min(max_gap, num_views - max_t_gap - 1)
#         min_gap = max(min(min_gap, max_gap), n_ctx - 1)

#         if max_gap < min_gap:
#             raise ValueError(
#                 f"Scene '{scene}' has only {num_views} frames, which is too "
#                 f"few for context_gap={min_gap} + target_gap={max_t_gap}."
#             )

#         # --- pick a random context window size ---
#         context_gap = torch.randint(
#             min_gap, max_gap + 1, size=(), device=device
#         ).item()

#         # --- pick a random starting position ---
#         # Need: start + context_gap + max_t_gap <= num_views - 1
#         max_start = num_views - context_gap - max_t_gap - 1
#         if self.stage == "test" or self.is_overfitting:
#             start = 0
#         else:
#             start = torch.randint(
#                 0, max(max_start, 0) + 1, size=(), device=device
#             ).item()

#         # --- select context views via farthest point sampling ---
#         ctx_window = torch.arange(start, start + context_gap + 1)
#         ctx_cam_pos = extrinsics[ctx_window, :3, 3].unsqueeze(0)  # (1, W, 3)
#         ctx_fps = farthest_point_sample(ctx_cam_pos, n_ctx).squeeze(0)
#         context_indices = ctx_window[ctx_fps].sort().values

#         # --- select target views via farthest point sampling ---
#         last_context = context_indices[-1].item()
#         tgt_window_start = last_context + min_t_gap
#         tgt_window_end = min(last_context + max_t_gap, num_views - 1)
#         tgt_window = torch.arange(tgt_window_start, tgt_window_end + 1)

#         if n_tgt >= tgt_window.numel():
#             # Window is smaller than requested targets; take all.
#             target_indices = tgt_window
#         else:
#             tgt_cam_pos = extrinsics[tgt_window, :3, 3].unsqueeze(0)
#             tgt_fps = farthest_point_sample(tgt_cam_pos, n_tgt).squeeze(0)
#             target_indices = tgt_window[tgt_fps].sort().values

#         return context_indices.to(torch.int64), target_indices.to(torch.int64)

#     @property
#     def num_context_views(self) -> int:
#         return self.cfg.num_context_views

#     @property
#     def num_target_views(self) -> int:
#         return self.cfg.num_target_views
