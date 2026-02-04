from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"] | Float[Tensor, "gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"] | Float[Tensor, "gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"] | Float[Tensor, "gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"] | Float[Tensor, " gaussian"]
    scales: Float[Tensor, "batch gaussian 3"] | Float[Tensor, "gaussian 3"]
    rotations: Float[Tensor, "batch gaussian 4"] | Float[Tensor, "gaussian 4"]
    # levels: Float[Tensor, "batch gaussian"]
