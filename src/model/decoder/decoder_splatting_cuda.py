from dataclasses import dataclass
from math import sqrt
from typing import Literal

import torch
from gsplat import rasterization
from jaxtyping import Float
from torch import Tensor

from ..types import Gaussians

# from .cuda_splatting import DepthRenderingMode, render_cuda
from .decoder import Decoder, DecoderOutput

DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


def sort_gaussians_by_position(
    xyzs: Float[Tensor, "batch N 3"],
    opacities: Float[Tensor, "batch N 1"],
    rotations: Float[Tensor, "batch N 4"],
    scales: Float[Tensor, "batch N 3"],
    features: Float[Tensor, "batch N C D"],
    covariances: Float[Tensor, "batch N 3 3"],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Sort Gaussians by 3D position with priority x > y > z (vectorized).

    Uses quantization-based lexicographic sorting for efficiency.
    All attributes are sorted consistently using the same indices.
    Fully vectorized across batch dimension for better GPU utilization.
    """
    B, N, _ = xyzs.shape

    # Normalize coordinates to [0, 1] range per batch item
    xyz_min = xyzs.min(dim=1, keepdim=True).values  # [B, 1, 3]
    xyz_max = xyzs.max(dim=1, keepdim=True).values  # [B, 1, 3]
    xyz_range = (xyz_max - xyz_min).clamp(min=1e-6)
    xyz_normalized = (xyzs - xyz_min) / xyz_range  # [B, N, 3] in [0, 1]

    # Create composite sort key: x has highest priority, then y, then z
    # Quantize to 21 bits per dimension (fits in int64)
    QUANT_LEVELS = 2097152  # 2^21
    x_quant = (xyz_normalized[..., 0] * (QUANT_LEVELS - 1)).long()  # [B, N]
    y_quant = (xyz_normalized[..., 1] * (QUANT_LEVELS - 1)).long()  # [B, N]
    z_quant = (xyz_normalized[..., 2] * (QUANT_LEVELS - 1)).long()  # [B, N]
    sort_key = (x_quant << 42) | (y_quant << 21) | z_quant  # [B, N]

    # Sort each batch independently
    sort_indices = torch.argsort(sort_key, dim=1)  # [B, N]

    # Helper function to gather along dim=1 for tensors with varying trailing dims
    def gather_sorted(tensor: Tensor, indices: Tensor) -> Tensor:
        # tensor: [B, N, ...], indices: [B, N]
        # Expand indices to match tensor's shape
        idx_shape = list(indices.shape) + [1] * (tensor.ndim - 2)
        idx_expanded = indices.view(*idx_shape).expand_as(tensor)
        return torch.gather(tensor, dim=1, index=idx_expanded)

    return (
        gather_sorted(xyzs, sort_indices),
        gather_sorted(opacities, sort_indices),
        gather_sorted(rotations, sort_indices),
        gather_sorted(scales, sort_indices),
        gather_sorted(features, sort_indices),
        gather_sorted(covariances, sort_indices),
    )


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color: list[float]
    make_scale_invariant: bool


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def rendering_fn(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
    ) -> DecoderOutput:
        B, V, _, _ = intrinsics.shape
        H, W = image_shape
        rendered_imgs, rendered_depths, rendered_alphas = [], [], []
        xyzs, opacitys, rotations, scales, features = (
            gaussians.means,
            gaussians.opacities,
            gaussians.rotations,
            gaussians.scales,
            gaussians.harmonics.permute(0, 1, 3, 2).contiguous(),
        )
        covariances = gaussians.covariances

        # # Sort Gaussians by 3D position (priority: x > y > z)
        # xyzs, opacitys, rotations, scales, features, covariances = sort_gaussians_by_position(
        #     xyzs, opacitys, rotations, scales, features, covariances
        # )
        for i in range(B):
            xyz_i = xyzs[i].float()
            feature_i = features[i].float()
            covar_i = covariances[i].float()
            scale_i = scales[i].float()
            rotation_i = rotations[i].float()
            opacity_i = opacitys[i].squeeze().float()
            test_w2c_i = extrinsics[i].float().inverse()  # (V, 4, 4)
            test_intr_i_normalized = intrinsics[i].float()
            # Denormalize the intrinsics into standred format
            test_intr_i = test_intr_i_normalized.clone()
            test_intr_i[:, 0] = test_intr_i_normalized[:, 0] * W
            test_intr_i[:, 1] = test_intr_i_normalized[:, 1] * H
            sh_degree = int(sqrt(feature_i.shape[-2])) - 1

            rendering_list = []
            rendering_depth_list = []
            rendering_alpha_list = []
            for j in range(V):
                rendering, alpha, _ = rasterization(
                    xyz_i,
                    rotation_i,
                    scale_i,
                    opacity_i,
                    feature_i,
                    test_w2c_i[j : j + 1],
                    test_intr_i[j : j + 1],
                    W,
                    H,
                    sh_degree=sh_degree,
                    # near_plane=near[i].mean(), far_plane=far[i].mean(),
                    render_mode="RGB+D",
                    packed=False,
                    near_plane=1e-10,
                    backgrounds=self.background_color.unsqueeze(0).repeat(1, 1),
                    radius_clip=0.1,
                    covars=covar_i,
                    rasterize_mode="classic",
                )  # (V, H, W, 3)
                rendering_img, rendering_depth = torch.split(rendering, [3, 1], dim=-1)
                rendering_img = rendering_img.clamp(0.0, 1.0)
                rendering_list.append(rendering_img.permute(0, 3, 1, 2))
                rendering_depth_list.append(rendering_depth)
                rendering_alpha_list.append(alpha)
            # squeeze(-1) removes only the trailing channel dim, preserving view dim when V=1
            rendered_depths.append(torch.cat(rendering_depth_list, dim=0).squeeze(-1))
            rendered_imgs.append(torch.cat(rendering_list, dim=0))
            rendered_alphas.append(torch.cat(rendering_alpha_list, dim=0).squeeze(-1))
        return DecoderOutput(
            torch.stack(rendered_imgs),
            torch.stack(rendered_depths),
            torch.stack(rendered_alphas),
            lod_rendering=None,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
    ) -> DecoderOutput:
        return self.rendering_fn(
            gaussians,
            extrinsics,
            intrinsics,
            near,
            far,
            image_shape,
            depth_mode,
            cam_rot_delta,
            cam_trans_delta,
        )
