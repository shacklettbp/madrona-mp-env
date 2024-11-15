import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
from flax.core import FrozenDict
from typing import Callable

import argparse
from functools import partial

import madrona_learn
from madrona_learn import (
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
    LayerNorm,
)
from madrona_learn.rnn import LSTM

from hash_encoder import HashGridEncoder

def assert_valid_input(tensor):
    #checkify.check(jnp.isnan(tensor).any() == False, "NaN!")
    #checkify.check(jnp.isinf(tensor).any() == False, "Inf!")
    return None
    #checkify.check(jnp.isnan(tensor).any() == False, "NaN!")
    #checkify.check(jnp.isinf(tensor).any() == False, "Inf!")

class PolicyRNN(nn.Module):
    rnn: nn.Module
    norm: nn.Module

    @staticmethod
    def create(num_hidden_channels, num_layers, dtype, rnn_cls = LSTM):
        return PolicyRNN(
            rnn = rnn_cls(
                num_hidden_channels = num_hidden_channels,
                num_layers = num_layers,
                dtype = dtype,
            ),
            norm = LayerNorm(dtype=dtype),
        )

    @nn.nowrap
    def init_recurrent_state(self, N):
        return self.rnn.init_recurrent_state(N)

    @nn.nowrap
    def clear_recurrent_state(self, rnn_states, should_clear):
        return self.rnn.clear_recurrent_state(rnn_states, should_clear)

    def setup(self):
        pass

    def __call__(
        self,
        cur_hiddens,
        x,
        train,
    ):
        team_features, agent_features = x

        out, new_hiddens = self.rnn(cur_hiddens, team_features, train)
        return (self.norm(out), agent_features), new_hiddens

    def sequence(
        self,
        start_hiddens,
        seq_ends,
        seq_x,
        train,
    ):
        team_features, agent_features = seq_x

        return (
            self.norm(self.rnn.sequence(
                start_hiddens, seq_ends, team_features, train)),
            agent_features,
        )


class TeamPrefixCommon(nn.Module):
    dtype: jnp.dtype
    num_embed_channels: int = 64
    embed_init: Callable = jax.nn.initializers.orthogonal()

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        jax.tree_map(lambda x: assert_valid_input(x), obs)

        def process_lidar(lidar):
            lidar = lidar.swapaxes(-2, -3)
            lidar = lidar.reshape(*lidar.shape[0:-2], -1)

            lidar = nn.Conv(
                    features=16,
                    kernel_size=(3,),
                    strides=(2,),
                    padding='SAME',
                    dtype=self.dtype,
                )(lidar)

            lidar = nn.leaky_relu(lidar)

            lidar = nn.Conv(
                    features=16,
                    kernel_size=(3,),
                    strides=(2,),
                    padding='SAME',
                    dtype=self.dtype,
                )(lidar)

            lidar = nn.leaky_relu(lidar)

            lidar = nn.Conv(
                    features=16,
                    kernel_size=(3,),
                    strides=(2,),
                    padding='SAME',
                    dtype=self.dtype,
                )(lidar)

            lidar = lidar.reshape(*lidar.shape[:-2], -1)
            lidar = LayerNorm(dtype=self.dtype)(lidar)
            lidar = nn.leaky_relu(lidar)

            return lidar

        def embed_team_agents(ob, name):
            o = nn.Dense(
                self.num_embed_channels,
                use_bias = True,
                kernel_init = self.embed_init,
                bias_init = jax.nn.initializers.constant(0),
                dtype = self.dtype,
                name = name,
            )(ob)

            o = LayerNorm(dtype=self.dtype)(o)
            o = nn.leaky_relu(o)
            return o

        obs, global_ob = obs.pop('full_team_global')

        obs, my_obs = obs.pop('full_team_players')
        obs, enemy_obs = obs.pop('full_team_enemies')
        obs, last_known_enemy_obs = obs.pop('full_team_last_known_enemies')

        obs, team_fwd_lidar = obs.pop('full_team_fwd_lidar')
        obs, team_rear_lidar = obs.pop('full_team_rear_lidar')

        global_features = nn.Dense(
            self.num_embed_channels,
            use_bias = True,
            kernel_init = self.embed_init,
            bias_init = jax.nn.initializers.constant(0),
            dtype = self.dtype,
            name = 'global_embed',
        )(global_ob)

        my_positions = my_obs[..., 8:11]
        enemy_positions = enemy_obs[..., 8:11]
        last_known_enemy_positions = last_known_enemy_obs[..., 8:11]

        enemy_mask = enemy_obs[..., [-1]]

        my_features = embed_team_agents(my_obs, 'my_embed')
        enemy_features = embed_team_agents(enemy_obs, 'enemy_embed')
        last_known_enemy_features = embed_team_agents(
                    last_known_enemy_obs, 'last_known_enemy_embed')

        team_fwd_lidar = process_lidar(team_fwd_lidar)
        team_rear_lidar = process_lidar(team_rear_lidar)

        team_lidar = jnp.concatenate([
                team_fwd_lidar,
                team_rear_lidar,
            ], axis=-1)

        return FrozenDict({
            'global_features': global_features,
            'my_features': my_features,
            'my_lidar': team_lidar,
            'enemy_features': enemy_features,
            'last_known_enemy_features': last_known_enemy_features,
            'my_positions': my_positions,
            'enemy_positions': enemy_positions,
            'last_known_enemy_positions': last_known_enemy_positions,
            'enemy_mask': enemy_mask,
        })

minimap_res = 16
@partial(jax.vmap, in_axes=0, out_axes=0)
def build_map(global_features,
              my_features,
              enemy_features,
              last_known_enemy_features,
              my_positions,
              enemy_positions,
              last_known_enemy_positions,
              enemy_masks):
    assert len(my_features.shape) == 2
    
    team_size = my_features.shape[-2]
    assert enemy_features.shape[-2] == team_size
    assert last_known_enemy_features.shape[-2] == team_size

    minimap = jnp.zeros(
        (minimap_res, minimap_res, my_features.shape[-1]), my_features.dtype)

    minimap = minimap.at[:, :].set(global_features)

    minimap_counts = jnp.ones((minimap_res, minimap_res, 1), jnp.int32)

    def discretize_pos(pos):
        i_x = (pos[0] * minimap_res).astype(jnp.int32)
        i_y = (pos[1] * minimap_res).astype(jnp.int32)

        i_x = jnp.clip(i_x, min=0, max=minimap_res-1)
        i_y = jnp.clip(i_y, min=0, max=minimap_res-1)

        return i_x, i_y

    def add_to_minimap(ob, pos, mask, minimap, minimap_counts):
        x_idx, y_idx = discretize_pos(pos)

        is_valid = jnp.logical_and(ob[0] == 1.0, mask == 1.0)

        count_add = jnp.where(
            is_valid, jnp.array(1, jnp.int32), jnp.array(0, jnp.int32))

        ob = jnp.where(is_valid, ob, jnp.zeros_like(ob))

        old_count = minimap_counts[y_idx, x_idx]
        new_count = old_count + count_add

        old_frac = old_count.astype(jnp.float32) / new_count.astype(jnp.float32)
        new_frac = 1.0 / new_count.astype(jnp.float32)

        minimap = minimap.at[y_idx, x_idx].set((
            old_frac * minimap[y_idx, x_idx].astype(jnp.float32) +
            new_frac * ob.astype(jnp.float32)).astype(minimap.dtype))

        minimap_counts = minimap_counts.at[y_idx, x_idx].set(new_count)

        return minimap, minimap_counts


    for i in range(team_size):
        my = my_features[i]
        enemy = enemy_features[i]
        last_known_enemy = last_known_enemy_features[i]

        my_pos = my_positions[i]
        enemy_pos = enemy_positions[i]
        last_known_enemy_pos = last_known_enemy_positions[i]

        enemy_mask = enemy_masks[i]

        minimap, minimap_counts = add_to_minimap(
            my, my_pos, 1.0, minimap, minimap_counts)

        minimap, minimap_counts = add_to_minimap(
            enemy, enemy_pos, enemy_mask, minimap, minimap_counts)

        minimap, minimap_counts = add_to_minimap(
            last_known_enemy, last_known_enemy_pos,
            1.0, minimap, minimap_counts)

    return minimap


def build_conv_backbone(minimap, dtype):
    o = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            dtype=dtype,
        )(minimap)

    o = nn.leaky_relu(o)
    o = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            dtype=dtype,
        )(o)

    o = nn.leaky_relu(o)
    o = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            dtype=dtype,
        )(o)

    o = o.reshape(*o.shape[:-3], -1)
    o = LayerNorm(dtype=dtype)(o)
    o = nn.leaky_relu(o)

    return o


class TeamActorNet(nn.Module):
    dtype: jnp.dtype
    use_minimap: bool

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        obs, global_features = obs.pop('global_features')
        obs, my_features = obs.pop('my_features')
        obs, my_lidar = obs.pop('my_lidar')
        obs, enemy_features = obs.pop('enemy_features')
        obs, last_known_enemy_features = obs.pop('last_known_enemy_features')
        obs, my_positions = obs.pop('my_positions')
        obs, enemy_positions = obs.pop('enemy_positions')
        obs, last_known_enemy_positions = obs.pop('last_known_enemy_positions')
        obs, enemy_masks = obs.pop('enemy_mask')


        num_batch_axes = len(my_features.shape) - 2


        map_encoded = build_map(
            global_features.reshape(
                -1, *global_features.shape[num_batch_axes:]),
            my_features.reshape(
                -1, *my_features.shape[num_batch_axes:]),
            enemy_features.reshape(
                -1, *enemy_features.shape[num_batch_axes:]),
            last_known_enemy_features.reshape(
                -1, *last_known_enemy_features.shape[num_batch_axes:]),
            my_positions.reshape(
                -1, *my_positions.shape[num_batch_axes:]),
            enemy_positions.reshape(
                -1, *enemy_positions.shape[num_batch_axes:]),
            last_known_enemy_positions.reshape(
                -1, *last_known_enemy_positions.shape[num_batch_axes:]),
            enemy_masks.reshape(
                -1, *enemy_masks.shape[num_batch_axes:]),
        )

        map_encoded = map_encoded.reshape(
            *my_features.shape[0:num_batch_axes], minimap_res, minimap_res,
            map_encoded.shape[-1])

        team_features = build_conv_backbone(map_encoded, self.dtype)

        #team_features = MLP(
        #    num_channels = 256,
        #    num_layers = 2,
        #    dtype = self.dtype,
        #)(team_features, train)

        agent_features = jnp.concatenate([
                my_features,
                my_lidar,
            ], axis=-1)

        return team_features, agent_features


class TeamCriticNet(nn.Module):
    dtype: jnp.dtype
    use_minimap: bool

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        obs, global_features = obs.pop('global_features')
        obs, my_features = obs.pop('my_features')
        obs, my_lidar = obs.pop('my_lidar')
        obs, enemy_features = obs.pop('enemy_features')
        obs, last_known_enemy_features = obs.pop('last_known_enemy_features')
        obs, my_positions = obs.pop('my_positions')
        obs, enemy_positions = obs.pop('enemy_positions')
        obs, last_known_enemy_positions = obs.pop('last_known_enemy_positions')
        obs, enemy_masks = obs.pop('enemy_mask')


        num_batch_axes = len(my_features.shape) - 2


        map_encoded = build_map(
            global_features.reshape(
                -1, *global_features.shape[num_batch_axes:]),
            my_features.reshape(
                -1, *my_features.shape[num_batch_axes:]),
            enemy_features.reshape(
                -1, *enemy_features.shape[num_batch_axes:]),
            last_known_enemy_features.reshape(
                -1, *last_known_enemy_features.shape[num_batch_axes:]),
            my_positions.reshape(
                -1, *my_positions.shape[num_batch_axes:]),
            enemy_positions.reshape(
                -1, *enemy_positions.shape[num_batch_axes:]),
            last_known_enemy_positions.reshape(
                -1, *last_known_enemy_positions.shape[num_batch_axes:]),
            jnp.ones_like(enemy_masks.reshape(
                -1, *enemy_masks.shape[num_batch_axes:])),
        )

        map_encoded = map_encoded.reshape(
            *my_features.shape[0:num_batch_axes], minimap_res, minimap_res,
            map_encoded.shape[-1])

        team_features = build_conv_backbone(map_encoded, self.dtype)

        print(team_features.shape)

        #team_features = MLP(
        #    num_channels = 256,
        #    num_layers = 2,
        #    dtype = self.dtype,
        #)(team_features, train)

        agent_features = jnp.concatenate([
                my_features,
                my_lidar,
            ], axis=-1)

        return team_features, agent_features

class TeamDiscreteActor(nn.Module):
    dtype: jnp.dtype
    embed_init: Callable = jax.nn.initializers.orthogonal()

    @nn.compact
    def __call__(self, input_features, train=False):
        @partial(jax.vmap, in_axes=(None, -2), out_axes=-2)
        def concat_features(team_features, agent_features):
            return jnp.concatenate([
                    team_features,
                    agent_features,
                ], axis=-1)

        team_features, agent_features = input_features

        features = concat_features(team_features, agent_features)

        features = nn.Dense(256,
                use_bias = True,
                kernel_init = self.embed_init,
                bias_init = jax.nn.initializers.constant(0),
                dtype = self.dtype,
                name = 'actor_merge',
            )(features)

        features = LayerNorm(dtype=self.dtype)(features)
        features = nn.leaky_relu(features)

        features = team_features[..., None, :] + features

        return DenseLayerDiscreteActor(
            actions_num_buckets = [4, 8, 5, 5, 2, 2, 3],
            dtype = self.dtype,
        )(features)

class TeamCritic(nn.Module):
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, input_features, train=False):
        team_features, agent_features = input_features
        return DenseLayerCritic(dtype=self.dtype)(team_features, train=train)

def make_policy(dtype, scene_name):
    prefix = TeamPrefixCommon(
        dtype = dtype,
    )

    actor_encoder = RecurrentBackboneEncoder(
        net = TeamActorNet(dtype, use_minimap=True),
        rnn = PolicyRNN.create(
            num_hidden_channels = 256,
            num_layers = 1,
            dtype = dtype,
        ),
    )

    critic_encoder = RecurrentBackboneEncoder(
        net = TeamCriticNet(dtype, use_minimap=True),
        rnn = PolicyRNN.create(
            num_hidden_channels = 256,
            num_layers = 1,
            dtype = dtype,
        ),
    )

    backbone = BackboneSeparate(
        prefix = prefix,
        actor_encoder = actor_encoder,
        critic_encoder = critic_encoder,
    )

    actor_critic = ActorCritic(
        backbone = backbone,
        actor = TeamDiscreteActor(dtype=dtype),
        critic = TeamCritic(dtype=dtype),
    )


    obs_preprocess = ObservationsEMANormalizer.create(
        decay = 0.99999,
        dtype = dtype,
        prep_fns = {
            'full_team_global': lambda x: x,
            'full_team_players': lambda x: x,
            'full_team_enemies': lambda x: x,
            'full_team_last_known_enemies': lambda x: x,
            'full_team_fwd_lidar': lambda x: x,
            'full_team_rear_lidar': lambda x: x,
        },
        skip_normalization = {
            'full_team_global',
            'full_team_players',
            'full_team_enemies',
            'full_team_last_known_enemies',
            'full_team_fwd_lidar',
            'full_team_rear_lidar',
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
