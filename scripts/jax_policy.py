import jax
import numpy as np
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
from flax.core import FrozenDict, frozen_dict
from typing import Callable

import argparse
from functools import partial

import madrona_learn
from madrona_learn import (
    DiscreteActionDistributions, ContinuousActionDistributions,
    DiscreteActionsConfig, ContinuousActionsConfig,
    ActorCritic, TrainConfig, PPOConfig,
    BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder,
    ObservationsEMANormalizer,
    Policy,
)

from madrona_learn.models import (
    MLP,
    EntitySelfAttentionNet,
    DenseLayerDiscreteActor,
    DenseLayerCritic,
    DreamerV3Critic,
    LayerNorm,
)
from madrona_learn.rnn import LSTM

from hash_encoder import HashGridEncoder

actions_config = {
    'discrete': DiscreteActionsConfig(
        actions_num_buckets = [ 3, 8, 3, 3 ],
    ),
    #'aim': ContinuousActionsConfig(
    #    stddev_min = 0.001,
    #    stddev_max = 1.0,
    #    num_dims = 2,
    #),
    'aim': DiscreteActionsConfig(
        actions_num_buckets = [ 13, 7 ],
    ),
}

def assert_valid_input(tensor):
    #checkify.check(jnp.isnan(tensor).any() == False, "NaN!")
    #checkify.check(jnp.isinf(tensor).any() == False, "Inf!")
    return None
    #checkify.check(jnp.isnan(tensor).any() == False, "NaN!")
    #checkify.check(jnp.isinf(tensor).any() == False, "Inf!")

def normalize_pos(pos):
    return pos

def vaswani_positional_embedding(embed_size, dtype):
    def embed(pos):
        def embed_dim(i, embedding):
            v = pos * (2.0 ** i) * jnp.pi
            sin_embedding = jnp.sin(v)
            cos_embedding = jnp.cos(v)

            embedding = embedding.at[..., 2 * i, :].set(
                sin_embedding.astype(dtype))
            embedding = embedding.at[..., 2 * i + 1, :].set(
                cos_embedding.astype(dtype))

            return embedding

        embedding = jnp.empty(
            (*pos.shape[:-1], embed_size, pos.shape[-1]), dtype=dtype)

        embedding = lax.fori_loop(0, embed_size // 2, embed_dim, embedding)

        return embedding.reshape(*embedding.shape[:-2], -1)
    return embed


class PrefixCommon(nn.Module):
    dtype: jnp.dtype
    num_embed_channels: int = 64
    embed_init: Callable = jax.nn.initializers.orthogonal(scale=np.sqrt(2))

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        jax.tree_map(lambda x: assert_valid_input(x), obs)

        obs, self_ob = obs.pop('self')
        obs, fwd_lidar = obs.pop('fwd_lidar')
        obs, rear_lidar = obs.pop('rear_lidar')

        #fwd_lidar = process_lidar(fwd_lidar)
        #rear_lidar = process_lidar(rear_lidar)

        fwd_lidar = nn.Dense(
                self.num_embed_channels,
                use_bias = True,
                kernel_init = self.embed_init,
                bias_init = jax.nn.initializers.constant(0),
                dtype = self.dtype,
                name = 'fwd_lidar_embed',
            )(fwd_lidar.reshape(*fwd_lidar.shape[:-3], -1))

        fwd_lidar = LayerNorm(dtype=self.dtype)(fwd_lidar)
        fwd_lidar = nn.leaky_relu(fwd_lidar)

        rear_lidar = nn.Dense(
                self.num_embed_channels,
                use_bias = True,
                kernel_init = self.embed_init,
                bias_init = jax.nn.initializers.constant(0),
                dtype = self.dtype,
                name = 'rear_lidar_embed',
            )(rear_lidar.reshape(*rear_lidar.shape[:-3], -1))

        rear_lidar = LayerNorm(dtype=self.dtype)(rear_lidar)
        rear_lidar = nn.leaky_relu(rear_lidar)

        obs, teammates = obs.pop('teammates')
        obs, opponents = obs.pop('opponents')
        obs, opponents_last_known = obs.pop('opponents_last_known')

        obs, self_pos = obs.pop('self_pos')
        obs, teammates_positions = obs.pop('teammate_positions')
        obs, opponents_positions = obs.pop('opponent_positions')
        obs, opponents_last_known_positions = obs.pop('opponent_last_known_positions')

        obs, opponent_masks = obs.pop('opponent_masks')
        obs, reward_coefs = obs.pop('reward_coefs')

        enc = vaswani_positional_embedding(16, self.dtype)

        self_pos_enc = enc(self_pos)

        self_features = jnp.concatenate([
                self_ob,
                reward_coefs,
                self_pos_enc.astype(self.dtype),
            ], axis=-1)

        self_features = nn.Dense(
            self.num_embed_channels,
            use_bias = True,
            kernel_init = self.embed_init,
            bias_init = jax.nn.initializers.constant(0),
            dtype = self.dtype,
            name = 'self_embed',
        )(self_features)

        self_features = LayerNorm(dtype=self.dtype)(self_features)
        self_features = nn.leaky_relu(self_features)

        teammates_features = nn.Dense(
            self.num_embed_channels,
            use_bias = True,
            kernel_init = self.embed_init,
            bias_init = jax.nn.initializers.constant(0),
            dtype = self.dtype,
            name = 'teammates_embed',
        )(teammates)
        teammates_features = LayerNorm(dtype=self.dtype)(teammates_features)
        teammates_features = nn.leaky_relu(teammates_features)

        opponents_features = nn.Dense(
            self.num_embed_channels,
            use_bias = True,
            kernel_init = self.embed_init,
            bias_init = jax.nn.initializers.constant(0),
            dtype = self.dtype,
            name = 'opponents_embed',
        )(opponents)
        opponents_features = LayerNorm(dtype=self.dtype)(opponents_features)
        opponents_features = nn.leaky_relu(opponents_features)

        opponents_last_known_features = nn.Dense(
            self.num_embed_channels,
            use_bias = True,
            kernel_init = self.embed_init,
            bias_init = jax.nn.initializers.constant(0),
            dtype = self.dtype,
            name = 'opponents_last_known_embed',
        )(opponents_last_known)
        opponents_last_known_features = LayerNorm(dtype=self.dtype)(
            opponents_last_known_features)
        opponents_last_known_features = nn.leaky_relu(opponents_last_known_features)

        return FrozenDict({
            'self': self_features,
            'fwd_lidar': fwd_lidar,
            'rear_lidar': rear_lidar,

            'teammates': teammates_features,
            'opponents': opponents_features,
            'opponents_last_known': opponents_last_known_features,

            'self_pos': self_pos,
            'teammates_positions': teammates_positions,
            'opponents_positions': opponents_positions,
            'opponents_last_known_positions': opponents_last_known_positions,

            'opponent_masks': opponent_masks,
        })


class MaxPoolNet(nn.Module):
    dtype: jnp.dtype
    num_embed_channels: int
    embed_init: Callable = jax.nn.initializers.orthogonal(scale=np.sqrt(2))

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        obs_vec = jnp.concatenate([
            obs['self'],
            obs['fwd_lidar'],
            obs['rear_lidar'],
            jnp.max(obs['teammates'], axis=-2),
            jnp.max(obs['opponents'], axis=-2),
            jnp.max(obs['opponents_last_known'], axis=-2),
        ], axis=-1)

        return MLP(
                num_channels = 512,
                num_layers = 3,
                dtype = self.dtype,
            )(obs_vec, train)

class ActorNet(nn.Module):
    dtype: jnp.dtype
    use_maxpool_net: bool

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        obs, opponents = obs.pop('opponents')

        opponents = jnp.where(obs['opponent_masks'] == 1.0, opponents, 0.0)

        obs = obs.copy({'opponents': opponents})

        if self.use_maxpool_net:
            return MaxPoolNet(
                dtype=self.dtype,
                num_embed_channels = 512,
            )(obs, train)
        else:
            #obs, self_ob = obs.pop('self')
            #obs, map_ob = obs.pop('map')

            #obs = obs.copy({
            #    'self': jnp.concatenate([self_ob, map_ob], axis=-1)
            #})

            return EntitySelfAttentionNet(
                    num_embed_channels = 128,
                    num_out_channels = 256,
                    num_heads = 4,
                    dtype = self.dtype,
                )(obs, train=train)


class CriticNet(nn.Module):
    dtype: jnp.dtype
    use_maxpool_net: bool

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
       return MaxPoolNet(
           dtype=self.dtype,
           num_embed_channels = 512,
       )(obs, train)


class ActorDistributions(flax.struct.PyTreeNode):
    discrete: DiscreteActionDistributions
    aim: DiscreteActionDistributions

    def sample(self, prng_key):
        discrete_rnd, aim_rnd = random.split(prng_key)
        
        discrete_actions, discrete_log_probs = self.discrete.sample(discrete_rnd)
        continuous_actions, continuous_log_probs = self.aim.sample(aim_rnd)

        return frozen_dict.freeze({
            'discrete': discrete_actions,
            'aim': continuous_actions,
        }), frozen_dict.freeze({
            'discrete': discrete_log_probs,
            'aim': continuous_log_probs,
        })

    def best(self):
        return frozen_dict.freeze({
            'discrete': self.discrete.best(),
            'aim': self.aim.best(),
        })

    def action_stats(self, actions):
        discrete_log_probs, discrete_entropies = self.discrete.action_stats(actions['discrete'])
        continuous_log_probs, continuous_entropies = self.aim.action_stats(actions['aim'])

        return frozen_dict.freeze({
            'discrete': discrete_log_probs,
            'aim': continuous_log_probs,
        }), frozen_dict.freeze({
            'discrete': discrete_entropies,
            'aim': continuous_entropies,
        })


class ActorHead(nn.Module):
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self,
        features,
        train=False,
    ):
        discrete_dist = DenseLayerDiscreteActor(
            cfg = actions_config['discrete'],
            dtype = self.dtype,
        )(features)

        aim_dist = DenseLayerDiscreteActor(
            cfg = actions_config['aim'],
            dtype = self.dtype,
        )(features)

        return ActorDistributions(
            discrete=discrete_dist,
            aim=aim_dist,
        )


def make_policy(dtype):
    prefix = PrefixCommon(
        dtype = dtype,
        num_embed_channels = 64,
    )

    actor_encoder = BackboneEncoder(
        net = ActorNet(dtype, use_maxpool_net=True),
    )

    critic_encoder = BackboneEncoder(
        net = CriticNet(dtype, use_maxpool_net=True),
    )

    backbone = BackboneSeparate(
        prefix = prefix,
        actor_encoder = actor_encoder,
        critic_encoder = critic_encoder,
    )

    actor_critic = ActorCritic(
        backbone = backbone,
        actor = ActorHead(
            dtype = dtype,
        ),
        #critic = DreamerV3Critic(dtype=dtype),
        critic = DenseLayerCritic(dtype=dtype),
    )

    obs_preprocess = ObservationsEMANormalizer.create(
        decay = 0.99999,
        dtype = dtype,
        prep_fns = {
            'filters_state': lambda x: x,
            'opponent_masks': lambda m: m,#m.astype(jnp.bool_),

            'self_pos': lambda x: x, #x.astype(dtype),
            'teammate_positions': lambda x: x, #x.astype(dtype),
            'opponent_positions': lambda x: x, #x.astype(dtype),
            'opponent_last_known_positions': lambda x: x,# x.astype(dtype),
        },
        skip_normalization = {
            'filters_state',
            'opponent_masks',

            'self_pos',
            'teammate_positions',
            'opponent_positions',
            'opponent_last_known_positions',
        },
    )

    def get_episode_scores(match_result):
        winner_id = match_result[..., 0]
        is_a_winner = winner_id == 0
        is_b_winner = winner_id == 1

        a_score = jnp.where(
            is_a_winner, 1, jnp.where(is_b_winner, 0, 0.5))
        b_score = 1 - a_score

        return jnp.stack((a_score, b_score))

    return Policy(
        actor_critic = actor_critic,
        obs_preprocess = obs_preprocess,
        get_episode_scores = get_episode_scores, 
    )
