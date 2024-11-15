import math

import chex
import flax.linen as nn
from flax.linen.dtypes import Dtype
import jax
import jax.numpy as jnp
import jax.random as jran
import functools
from typing import Any, Dict, Hashable, Iterable, Sequence, get_args

def empty_impl(clz):
    if "__dataclass_fields__" not in clz.__dict__:
        raise TypeError("class `{}` is not a dataclass".format(clz.__name__))

    fields = clz.__dict__["__dataclass_fields__"]

    def empty_fn(cls, /, **kwargs):
        """
        Create an empty instance of the given class, with untransformed fields set to given values.
        """
        for field_name, annotation in fields.items():
            if field_name not in kwargs:
                kwargs[field_name] = getattr(annotation.type, "empty", lambda: None)()
        return cls(**kwargs)

    setattr(clz, "empty", classmethod(empty_fn))
    return clz

# NOTE:
#   Jitting a vmapped function seems to give the desired performance boost, while vmapping a jitted
#   function might not work at all.  Except for the experiments I conducted myself, some related
#   issues:
# REF:
#   * <https://github.com/google/jax/issues/6312>
#   * <https://github.com/google/jax/issues/7449>
def vmap_jaxfn_with(
        # kwargs copied from `jax.vmap` source
        in_axes: int | Sequence[Any]=0,
        out_axes: Any = 0,
        axis_name: Hashable | None = None,
        axis_size: int | None = None,
        spmd_axis_name: Hashable | None = None,
    ):
    return functools.partial(
        jax.vmap,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
        spmd_axis_name=spmd_axis_name,
    )


def jit_jaxfn_with(
        # kwargs copied from `jax.jit` source
        static_argnums: int | Iterable[int] | None = None,
        static_argnames: str | Iterable[str] | None = None,
        device = None,
        backend: str | None = None,
        donate_argnums: int | Iterable[int] = (),
        inline: bool = False,
        keep_unused: bool = False,
        abstracted_axes: Any | None = None,
    ):
    return functools.partial(
        jax.jit,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        device=device,
        backend=backend,
        donate_argnums=donate_argnums,
        inline=inline,
        keep_unused=keep_unused,
        abstracted_axes=abstracted_axes,
    )


def next_multiple(value: int, step: int) -> int:
    return (value + step - 1) // step * step

cell_vert_offsets = {
    2: jnp.asarray([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.],
    ]),
    3: jnp.asarray([
        [0., 0., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 1., 1.],
        [1., 0., 0.],
        [1., 0., 1.],
        [1., 1., 0.],
        [1., 1., 1.],
    ]),
}

adjacent_offsets = {
    2: jnp.asarray([
        [0., 1.],
        [1., 0.],
        [0., -1.],
        [-1., 0.],
    ]),
    3: jnp.asarray([
        [0., 0., 1.],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 0., -1.],
        [0., -1., 0.],
        [-1., 0., 0.],
    ]),
}


class Encoder(nn.Module): ...


# TODO: enforce types used in arrays
@empty_impl
class HashGridEncoder(Encoder):
    # Let's use the same notations as in the paper

    # Number of levels (16).
    L: int
    # Maximum entries per level (hash table size) (2**14 to 2**24).
    T: int
    # Number of feature dimensions per entry (2).
    F: int
    # Coarsest resolution (16).
    N_min: int
    # Finest resolution (512 to 524288).
    N_max: int

    tv_scale: float

    param_dtype: Dtype = jnp.float32

    @property
    def b(self) -> float:
        # Equation(3)
        # Essentially, it is $(n_max / n_min) ** (1/(L - 1))$
        return math.exp((math.log(self.N_max) - math.log(self.N_min)) / (self.L - 1))

    @nn.compact
    def __call__(self, pos: jax.Array, bound: float) -> jax.Array:
        dim = pos.shape[-1]

        # CAVEAT: hashgrid encoder is defined only in the unit cube [0, 1)^3
        pos = (pos + bound) / (2 * bound)

        scales, resolutions, first_hash_level, offsets = [], [], 0, [0]
        for i in range(self.L):
            scale = self.N_min * (self.b**i) - 1
            scales.append(scale)
            res = math.ceil(scale) + 1
            resolutions.append(res)

            n_entries = next_multiple(res ** dim, 8)

            if n_entries <= self.T:
                first_hash_level += 1
            else:
                n_entries = self.T

            offsets.append(offsets[-1] + n_entries)

        latents = self.param(
            "latent codes stored on grid vertices",
            # paper:
            #   We initialize the hash table entries using the uniform distribution U(−10^{−4}, 10^{−4})
            #   to provide a small amount of randomness while encouraging initial predictions close
            #   to zero.
            lambda key, shape, dtype: jran.uniform(key, shape, dtype, -1e-4, 1e-4),
            (offsets[-1], self.F),
            self.param_dtype,
        )

        @jax.vmap
        @jax.vmap
        def make_vert_pos(pos_scaled: jax.Array):
            # [dim]
            pos_floored = jnp.floor(pos_scaled)
            # [2**dim, dim]
            vert_pos = pos_floored[None, :] + cell_vert_offsets[dim]
            return vert_pos.astype(jnp.uint32)

        @jax.vmap
        @jax.vmap
        def make_adjacent_pos(pos_scaled: jax.Array):
            # [dim]
            pos_floored = jnp.floor(pos_scaled)
            # [dim * 2, dim]
            adjacent_pos = pos_floored[None, :] + adjacent_offsets[dim]
            return adjacent_pos.astype(jnp.uint32)

        @vmap_jaxfn_with(in_axes=(0, 0))
        @vmap_jaxfn_with(in_axes=(None, 0))
        def make_tiled_indices(res, vert_pos):
            """(first 2 axes `[L, n_points]` are vmapped away)
            Inputs:
                res `uint32` `[L]`: each hierarchy's resolution
                vert_pos `uint32` `[L, n_points, B, dim]`: integer positions of grid cell's
                                                           vertices, of each level

            Returns:
                indices `uint32` `[L, n_points, B]`: grid cell indices of the vertices
            """
            # [dim]
            if dim == 2:
                strides = jnp.stack([jnp.ones_like(res), res]).T
            elif dim == 3:
                strides = jnp.stack([jnp.ones_like(res), res, res ** 2]).T
            else:
                raise NotImplementedError("{} is only implemented for 2D and 3D data".format(__class__.__name__))
            # [2**dim]
            indices = jnp.sum(strides[None, :] * vert_pos, axis=-1)
            return indices

        @jax.vmap
        @jax.vmap
        def make_hash_indices(vert_pos):
            """(first 2 axes `[L, n_points]` are vmapped away)
            Inputs:
                vert_pos `uint32` `[L, n_points, B, dim]`: integer positions of grid cell's
                                                           vertices, of each level

            Returns:
                indices `uint32` `[L, n_points, B]`: grid cell indices of the vertices
            """
            # use primes as reported in the paper
            primes = jnp.asarray([1, 2_654_435_761, 805_459_861], dtype=jnp.uint32)
            # [2**dim]
            if dim == 2:
                indices = vert_pos[:, 0] ^ (vert_pos[:, 1] * primes[1])
            elif dim == 3:
                indices = vert_pos[:, 0] ^ (vert_pos[:, 1] * primes[1]) ^ (vert_pos[:, 2] * primes[2])
            else:
                raise NotImplementedError("{} is only implemented for 2D and 3D data".format(__class__.__name__))
            return indices

        def make_indices(vert_pos, resolutions, first_hash_level):
            if first_hash_level > 0:
                resolutions = jnp.asarray(resolutions, dtype=jnp.uint32)
                indices = make_tiled_indices(resolutions[:first_hash_level], vert_pos[:first_hash_level, ...])
            else:
                indices = jnp.empty(0, dtype=jnp.uint32)
            if first_hash_level < self.L:
                indices = jnp.concatenate([indices, make_hash_indices(vert_pos[first_hash_level:, ...])], axis=0)
            indices = jnp.mod(indices, self.T)
            indices += jnp.asarray(offsets[:-1], dtype=jnp.uint32)[:, None, None]
            return indices

        @jax.vmap
        @jax.vmap
        def lerp_weights(pos_scaled: jax.Array):
            """(first 2 axes `[L, n_points]` are vmapped away)
            Inputs:
                pos_scaled `float` `[L, n_points, dim]`: coordinates of query points, scaled to the
                                                         hierarchy in question

            Returns:
                weights `float` `[L, n_points, 2**dim]`: linear interpolation weights for each cell
                                                         vertex
            """
            # [dim]
            pos_offset, _ = jnp.modf(pos_scaled)
            # [2**dim, dim]
            widths = jnp.clip(
                # cell_vert_offsets: [2**dim, dim]
                (1 - cell_vert_offsets[dim]) + (2 * cell_vert_offsets[dim] - 1) * pos_offset[None, :],
                0,
                1,
            )
            # [2**dim]
            return jnp.prod(widths, axis=-1)

        # [L]
        scales = jnp.asarray(scales, dtype=jnp.float32)
        # [L, n_points, dim]
        pos_scaled = pos[None, :, :] * scales[:, None, None] + 0.5
        # [L, n_points, 2**dim, dim]
        vert_pos = make_vert_pos(pos_scaled)

        # [L, n_points, 2**dim]
        indices = make_indices(vert_pos, resolutions, first_hash_level)

        # [L, n_points, 2**dim, F]
        vert_latents = latents[indices]
        # [L, n_points, 2**dim]
        vert_weights = lerp_weights(pos_scaled)

        # [L, n_points, F]
        encodings = (vert_latents * vert_weights[..., None]).sum(axis=-2)
        # [n_points, L*F]
        encodings = encodings.transpose(1, 0, 2).reshape(-1, self.L * self.F)

        ## Total variation
        if self.tv_scale > 0:
            # [L, n_points, dim * 2, dim]
            adjacent_pos = make_adjacent_pos(pos_scaled)

            # [L, n_points, dim * 2]
            adjacent_indices = make_indices(adjacent_pos, resolutions, first_hash_level)

            # [L, n_points, dim * 2, F]
            adjacent_latents = latents[adjacent_indices]

            # [L, n_points, dim * 2, F]
            tv = self.tv_scale * jnp.square(adjacent_latents - vert_latents[:, :, :1, :])

            # [L, n_points]
            tv = tv.sum(axis=(-2, -1))

            tv = tv.mean()
        else:
            tv = 0

        return encodings, tv
