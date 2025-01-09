import jax
import numpy as np
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
    DreamerV3Critic,
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
        out, new_hiddens = self.rnn(cur_hiddens, x, train)
        return self.norm(out), new_hiddens

    def sequence(
        self,
        start_hiddens,
        seq_ends,
        seq_x,
        train,
    ):
        return self.norm(
            self.rnn.sequence(start_hiddens, seq_ends, seq_x, train))


class HashGridPrefixCommon(nn.Module):
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        jax.tree_map(lambda x: assert_valid_input(x), obs)

        enc = HashGridEncoder(L=16, T=2**14, F=2, N_min=16, N_max=1024,
                              tv_scale=0, param_dtype=jnp.float32)

        obs, self_pos_ob = obs.pop('self_pos')

        encoded_self_pos, _ = enc(self_pos_ob, 1.0)

        obs, teammate_positions = obs.pop('teammate_positions')
        obs, opponent_positions = obs.pop('opponent_positions')
        obs, opponent_last_positions = obs.pop('opponent_last_known_positions')

        teammate_positions_shape = teammate_positions.shape
        opponent_positions_shape = opponent_positions.shape
        opponent_last_positions_shape = opponent_last_positions.shape

        teammate_positions = teammate_positions.reshape(
            -1, *teammate_positions.shape[2:])

        opponent_positions = opponent_positions.reshape(
            -1, *opponent_positions.shape[2:])

        opponent_last_positions = opponent_last_positions.reshape(
            -1, *opponent_last_positions.shape[2:])

        encoded_teammate_positions, _ = enc(teammate_positions, 1.0)
        encoded_opponent_positions, _ = enc(opponent_positions, 1.0)
        encoded_opponent_last_positions, _ = enc(opponent_last_positions, 1.0)

        encoded_teammate_positions = encoded_teammate_positions.reshape(
            *teammate_positions_shape[:-1], -1)

        encoded_opponent_positions = encoded_opponent_positions.reshape(
            *opponent_positions_shape[:-1], -1)

        encoded_opponent_last_positions = encoded_opponent_last_positions.reshape(
                *opponent_last_positions_shape[:-1], -1)

        encoded_self_pos = encoded_self_pos.astype(self.dtype)
        encoded_teammate_positions = encoded_teammate_positions.astype(self.dtype)
        encoded_opponent_positions = encoded_opponent_positions.astype(self.dtype)
        encoded_opponent_last_positions = encoded_opponent_last_positions.astype(self.dtype)

        obs, self_ob = obs.pop('self')
        obs, fwd_lidar = obs.pop('fwd_lidar')
        obs, rear_lidar = obs.pop('rear_lidar')

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


        fwd_lidar = process_lidar(fwd_lidar)
        rear_lidar = process_lidar(rear_lidar)

        self_ob = jnp.concatenate([
                encoded_self_pos,
                self_ob,
                fwd_lidar,
                rear_lidar,
            ], axis=-1)
        
        obs, teammates = obs.pop('teammates')

        obs, opponents = obs.pop('opponents')

        obs, opponents_last_known = obs.pop('opponents_last_known')

        obs, opponent_masks = obs.pop('opponent_masks')

        obs, agent_map = obs.pop('agent_map')
        obs, unmasked_agent_map = obs.pop('unmasked_agent_map')
        
        #assert len(obs) == 0

        teammates = jnp.concatenate([
            encoded_teammate_positions,
            teammates,
        ], axis=-1)

        opponents = jnp.concatenate([
            encoded_opponent_positions,
            opponents,
        ], axis=-1)

        opponents_last_known = jnp.concatenate([
            encoded_opponent_last_positions,
            opponents_last_known
        ], axis=-1)

        return FrozenDict({
            'self': self_ob,
            'teammates': teammates,
            'opponents': opponents,
            'opponents_last_known': opponents_last_known,
            'opponent_masks': opponent_masks,
            #'zones': zone_obs,
            'agent_map': agent_map,
            'unmasked_agent_map': unmasked_agent_map,
        })


minimap_res = 16
@partial(jax.vmap, in_axes=0, out_axes=0)
def build_map(self_features,
              teammates_features,
              opponents_features,
              last_known_opponents_features,
              self_position,
              teammates_positions,
              opponents_positions,
              last_known_opponents_positions,
              opponent_masks):
    assert len(self_features.shape) == 1
    
    num_teammates = teammates_features.shape[-2]
    num_opponents = opponents_features.shape[-2]

    minimap = jnp.zeros(
        (minimap_res, minimap_res, self_features.shape[-1]),
        self_features.dtype)

    minimap = minimap.at[:, :].set(self_features)

    minimap_counts = jnp.ones((minimap_res, minimap_res, 1), jnp.int32)

    def discretize_pos(pos):
        i_x = (pos[0] * minimap_res).astype(jnp.int32)
        i_y = (pos[1] * minimap_res).astype(jnp.int32)

        i_x = jnp.clip(i_x, min=0, max=minimap_res-1)
        i_y = jnp.clip(i_y, min=0, max=minimap_res-1)

        return i_x, i_y

    def add_to_minimap(ob, pos, mask, minimap, minimap_counts):
        x_idx, y_idx = discretize_pos(pos)

        is_valid = jnp.logical_and(pos[0] != 1000.0, mask == 1.0)

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

    minimap, minimap_counts = add_to_minimap(
        self_features, self_position, 1.0,
        minimap, minimap_counts)

    for i in range(num_teammates):
        teammate_features = teammates_features[i]
        teammate_pos = teammates_positions[i]

        minimap, minimap_counts = add_to_minimap(
            teammate_features, teammate_pos, 1.0,
            minimap, minimap_counts)

    for i in range(num_opponents):
        opponent_features = opponents_features[i]
        opponent_pos = opponents_positions[i]

        minimap, minimap_counts = add_to_minimap(
            opponent_features, opponent_pos, opponent_masks[i],
            minimap, minimap_counts)

        last_known_opponent_features = last_known_opponents_features[i]
        last_known_opponent_pos = last_known_opponents_positions[i]

        minimap, minimap_counts = add_to_minimap(
            last_known_opponent_features, last_known_opponent_pos, 1.0,
            minimap, minimap_counts)

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

        def process_lidar(lidar):
            lidar = lidar.swapaxes(-2, -3)
            lidar = lidar.reshape(*lidar.shape[0:-2], -1)

            lidar = nn.Conv(
                    features=16,
                    kernel_size=(3,),
                    strides=(2,),
                    padding='SAME',
                    dtype=self.dtype,
                    kernel_init=self.embed_init,
                )(lidar)

            lidar = nn.leaky_relu(lidar)

            lidar = nn.Conv(
                    features=16,
                    kernel_size=(3,),
                    strides=(2,),
                    padding='SAME',
                    dtype=self.dtype,
                    kernel_init=self.embed_init,
                )(lidar)

            lidar = nn.leaky_relu(lidar)

            lidar = nn.Conv(
                    features=16,
                    kernel_size=(3,),
                    strides=(2,),
                    padding='SAME',
                    dtype=self.dtype,
                    kernel_init=self.embed_init,
                )(lidar)

            lidar = lidar.reshape(*lidar.shape[:-2], -1)
            lidar = LayerNorm(dtype=self.dtype)(lidar)
            lidar = nn.leaky_relu(lidar)

            return lidar

        obs, self_ob = obs.pop('self')
        obs, fwd_lidar = obs.pop('fwd_lidar')
        obs, rear_lidar = obs.pop('rear_lidar')

        fwd_lidar = process_lidar(fwd_lidar)
        rear_lidar = process_lidar(rear_lidar)

        obs, teammates = obs.pop('teammates')
        obs, opponents = obs.pop('opponents')
        obs, opponents_last_known = obs.pop('opponents_last_known')


        obs, self_pos = obs.pop('self_pos')
        obs, teammates_positions = obs.pop('teammate_positions')
        obs, opponents_positions = obs.pop('opponent_positions')
        obs, opponents_last_known_positions = obs.pop('opponent_last_known_positions')

        obs, opponent_masks = obs.pop('opponent_masks')

        enc = HashGridEncoder(L=16, T=2**14, F=2, N_min=16, N_max=1024,
                              tv_scale=0, param_dtype=jnp.float32)

        self_pos_enc, _ = enc(self_pos, 1.0)

        self_features = jnp.concatenate([
                self_ob,
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
        #num_batch_dims = len(obs['self'].shape) - 1
        #num_embed_channels = self.num_embed_channels
        #embed_init = self.embed_init
        #dtype = self.dtype

        #def make_embed(name):
        #    def embed(x):
        #        o = nn.Dense(
        #            num_embed_channels,
        #            use_bias = True,
        #            kernel_init = embed_init,
        #            bias_init = jax.nn.initializers.constant(0),
        #            dtype = dtype,
        #            name = name,
        #        )(x)

        #        o = LayerNorm(dtype=self.dtype)(o)
        #        o = nn.leaky_relu(o)

        #        return o

        #    return embed

        #obs, self = obs.pop('self')
        #obs, teammates = obs.pop('teammates')
        #obs, opponents = obs.pop('opponents')
        #obs, opponents_last_known = obs.pop('opponents_last_known')
        ##obs, zones = obs.pop('zones')
        ##obs, map_data = obs.pop('map')

        #self_embed = make_embed('self_embed')(self)
        #teammates_embed = make_embed('teammates_embed')(teammates)
        #opponents_embed = make_embed('opponents_embed')(opponents)
        #opponents_last_known_embed = make_embed('opponents_last_known_embed')(
        #    opponents_last_known)
        ##zones_embed = make_embed('zones_embed')(zones)
        ##map_embed = make_embed('map_embed')(map_data)

        #teammates_embed = jnp.max(teammates_embed, axis=-2)
        #opponents_embed = jnp.max(opponents_embed, axis=-2)
        #opponents_last_known_embed = jnp.max(opponents_last_known_embed, axis=-2)
        ##zones_embed = jnp.max(zones_embed, axis=-2)

        ##assert len(obs) == 0

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
        #minimap = build_map(obs['self'],
        #                    obs['teammates'],
        #                    obs['opponents'],
        #                    obs['opponents_last_known'],
        #                    obs['self_pos'],
        #                    obs['teammates_positions'],
        #                    obs['opponents_positions'],
        #                    obs['opponents_last_known_positions'],
        #                    obs['opponent_masks'])


        #minimap_encoded = build_conv_backbone(minimap, self.dtype)

        #encoded_features = jnp.concatenate([
        #        minimap_encoded,
        #        obs['self'],
        #    ], axis=-1)

        #return MLP(
        #        num_channels = 256,
        #        num_layers = 2,
        #        dtype = self.dtype,
        #    )(encoded_features, train)

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
        #minimap = build_map(obs['self'],
        #                    obs['teammates'],
        #                    obs['opponents'],
        #                    obs['opponents_last_known'],
        #                    obs['self_pos'],
        #                    obs['teammates_positions'],
        #                    obs['opponents_positions'],
        #                    obs['opponents_last_known_positions'],
        #                    jnp.ones_like(obs['opponent_masks']))


        #minimap_encoded = build_conv_backbone(minimap, self.dtype)

        #encoded_features = jnp.concatenate([
        #        minimap_encoded,
        #        obs['self'],
        #    ], axis=-1)

        #return MLP(
        #        num_channels = 256,
        #        num_layers = 2,
        #        dtype = self.dtype,
        #    )(encoded_features, train)

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

class MapActorNet(nn.Module):
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        #lidar = nn.Conv(
        #        features=1,
        #        kernel_size=(lidar.shape[-2],),
        #        padding='CIRCULAR',
        #        dtype=self.dtype,
        #    )(lidar)

        obs, self_ob = obs.pop('self')
        obs, agent_map = obs.pop('agent_map')

        map_encoded = nn.Conv(
                features=32,
                kernel_size=(3, 3),
                strides=(2, 2),
                dtype=self.dtype,
            )(agent_map)

        map_encoded = nn.leaky_relu(map_encoded)
        map_encoded = nn.Conv(
                features=64,
                kernel_size=(3, 3),
                strides=(2, 2),
                dtype=self.dtype,
            )(map_encoded)

        map_encoded = nn.leaky_relu(map_encoded)
        map_encoded = nn.Conv(
                features=64,
                kernel_size=(3, 3),
                strides=(2, 2),
                dtype=self.dtype,
            )(map_encoded)

        map_encoded = map_encoded.reshape(*map_encoded.shape[:-3], -1)
        map_encoded = LayerNorm(dtype=self.dtype)(map_encoded)
        map_encoded = nn.leaky_relu(map_encoded)

        obs_vec = jnp.concatenate([
                self_ob, 
                map_encoded,
            ], axis=-1)

        return MLP(
            num_channels = 256,
            num_layers = 2,
            dtype = self.dtype,
        )(obs_vec, train)


class MapCriticNet(nn.Module):
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self,
        obs,
        train,
    ):
        #lidar = nn.Conv(
        #        features=1,
        #        kernel_size=(lidar.shape[-2],),
        #        padding='CIRCULAR',
        #        dtype=self.dtype,
        #    )(lidar)

        obs, self_ob = obs.pop('self')
        obs, agent_map = obs.pop('unmasked_agent_map')

        map_encoded = nn.Conv(
                features=32,
                kernel_size=(3, 3),
                strides=(2, 2),
                dtype=self.dtype,
            )(agent_map)

        map_encoded = nn.leaky_relu(map_encoded)
        map_encoded = nn.Conv(
                features=64,
                kernel_size=(3, 3),
                strides=(2, 2),
                dtype=self.dtype,
            )(map_encoded)

        map_encoded = nn.leaky_relu(map_encoded)
        map_encoded = nn.Conv(
                features=64,
                kernel_size=(3, 3),
                strides=(2, 2),
                dtype=self.dtype,
            )(map_encoded)

        map_encoded = map_encoded.reshape(*map_encoded.shape[:-3], -1)
        map_encoded = LayerNorm(dtype=self.dtype)(map_encoded)
        map_encoded = nn.leaky_relu(map_encoded)

        obs_vec = jnp.concatenate([
                self_ob, 
                map_encoded,
            ], axis=-1)

        return MLP(
            num_channels = 256,
            num_layers = 2,
            dtype = self.dtype,
        )(obs_vec, train)


def make_policy(dtype, scene_name, actions_cfg):
    use_map_net = False

    if use_map_net:
        prefix = PrefixCommon(
            dtype = dtype,
            num_embed_channels = 32,
        )

        actor_encoder = RecurrentBackboneEncoder(
            net = MapActorNet(dtype),
            rnn = PolicyRNN.create(
                num_hidden_channels = 512,
                num_layers = 1,
                dtype = dtype,
            ),
        )

        critic_encoder = RecurrentBackboneEncoder(
            net = MapCriticNet(dtype),
            rnn = PolicyRNN.create(
                num_hidden_channels = 512,
                num_layers = 1,
                dtype = dtype,
            ),
        )
    else:
        prefix = PrefixCommon(
            dtype = dtype,
            num_embed_channels = 64,
        )

        actor_encoder = RecurrentBackboneEncoder(
            net = ActorNet(dtype, use_maxpool_net=True),
            rnn = PolicyRNN.create(
                num_hidden_channels = 512,
                num_layers = 1,
                dtype = dtype,
            ),
        )

        critic_encoder = RecurrentBackboneEncoder(
            net = CriticNet(dtype, use_maxpool_net=True),
            rnn = PolicyRNN.create(
                num_hidden_channels = 512,
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
        actor = DenseLayerDiscreteActor(
            cfg = actions_cfg,
            dtype = dtype,
        ),
        #critic = DreamerV3Critic(dtype=dtype),
        critic = DenseLayerCritic(dtype=dtype),
    )


    def normalize_pos(pos):
        return pos

    def vaswani_positional_embedding(embed_size):
        def embed(pos):
            pos = normalize_pos(pos)

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

            return embedding
        return embed

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
