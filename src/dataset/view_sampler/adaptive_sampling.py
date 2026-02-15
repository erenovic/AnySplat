import torch
from jaxtyping import Float
from torch import Tensor


@torch.no_grad()
def pose_distance(
    reference_pose: Float[Tensor, "ref 4 4"],
    measurement_pose: Float[Tensor, "meas 4 4"],
) -> tuple[
    Float[Tensor, "ref meas"],
    Float[Tensor, "ref meas"],
    Float[Tensor, "ref meas"],
]:
    """Pairwise SE(3) pose distance between reference and measurement poses.

    Adapted from deep-video-mvs.  Expects camera-to-world (c2w) matrices,
    i.e. the translation column is the camera position in world space.

    Parameters
    ----------
    reference_pose   : (R, 4, 4) c2w matrices
    measurement_pose : (M, 4, 4) c2w matrices

    Returns
    -------
    combined_measure : (R, M) sqrt(t_measure**2 + R_measure**2)  — raw, not scene-normalised
    R_measure        : (R, M) geodesic-like rotation distance in [0, sqrt(2)]
    t_measure        : (R, M) ||Δcam_pos|| = Euclidean camera-position distance
    """
    # rel_pose[i, j] = T_ref[i]^{-1} @ T_meas[j]
    rel_pose = (
        reference_pose.inverse().unsqueeze(1)  # (R, 1, 4, 4)
        @ measurement_pose.unsqueeze(0)  # (1, M, 4, 4)
    )  # (R, M, 4, 4)

    R = rel_pose[:, :, :3, :3]
    t = rel_pose[:, :, :3, 3]

    # Rotation distance: sqrt(2 * (1 - trace(R) / 3)), in [0, sqrt(2)]
    R_trace = R[:, :, 0, 0] + R[:, :, 1, 1] + R[:, :, 2, 2]
    R_measure = (2.0 * (1.0 - (R_trace / 3.0).clamp(max=1.0))).clamp(min=0.0).sqrt()

    # For c2w matrices, ||t_rel|| = ||cam_meas - cam_ref|| (camera-position distance).
    t_measure = t.norm(dim=-1)

    combined_measure = (t_measure**2 + R_measure**2).sqrt()
    return combined_measure, R_measure, t_measure


@torch.no_grad()
def sample_views(
    extrinsics: Float[Tensor, "view 4 4"],
    context_indices: list[int],
) -> Tensor:
    """Return indices of all views compatible with the given context frames.

    Adapted from deep-video-mvs (adaptive-threshold neighbourhood filter).
    A view v is included if, for *every* context frame c_i, both its combined
    and translation distance to v are ≤ the maximum pairwise distance among
    the context frames themselves.

    This retains views that are "nearby" all context frames simultaneously,
    discarding frames that are far from any context frame.

    Returns a 1-D int64 tensor of compatible view indices (context frames
    are always included).
    """
    ctx_idx = torch.tensor(context_indices, dtype=torch.long, device=extrinsics.device)
    combined, _, t_measure = pose_distance(extrinsics[ctx_idx], extrinsics)
    # combined, t_measure: (n_ctx, N)

    # Upper thresholds: max distance from context frame i to any other context frame.
    combined_thresh = combined[:, ctx_idx].max(dim=1, keepdim=True).values  # (n_ctx, 1)
    t_thresh = t_measure[:, ctx_idx].max(dim=1, keepdim=True).values  # (n_ctx, 1)

    mask = ((combined <= combined_thresh) & (t_measure <= t_thresh)).all(dim=0)
    return mask.nonzero(as_tuple=True)[0]
