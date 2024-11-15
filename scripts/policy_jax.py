from madrona_learn import (
    ActorCritic, DiscreteActor, Critic, 
    BackboneShared, BackboneSeparate,
    BackboneEncoder, RecurrentBackboneEncoder,
)

from madrona_learn.models import (
    MLP, LinearLayerDiscreteActor, LinearLayerCritic,
)

from madrona_learn.rnn import LSTM, FastLSTM
from madrona_learn.moving_avg import EMANormalizer

import math
import torch
import torch.nn as nn

def assert_valid_input(tensor):
    assert(not torch.isnan(tensor).any())
    assert(not torch.isinf(tensor).any())

def setup_explore_obs(sim):
    self_obs_tensor = sim.self_observation_tensor().to_torch()
    lidar_tensor = sim.lidar_tensor().to_torch()

    N, A = self_obs_tensor.shape[0:2]
    batch_size = N * A

    obs_tensors = [
        self_obs_tensor.view(batch_size, *self_obs_tensor.shape[2:]),
        lidar_tensor.view(batch_size, *lidar_tensor.shape[2:]),
    ]

    def process_obs(self_obs, lidar):
        assert_valid_input(self_obs)
        assert_valid_input(lidar)
    
        return torch.cat([
            self_obs.view(self_obs.shape[0], -1),
            lidar,
        ], dim=1)

    num_obs_features = 0
    for tensor in obs_tensors:
        num_obs_features += math.prod(tensor.shape[1:])

    return obs_tensors, process_obs, num_obs_features


def make_explore_policy(process_obs, num_obs_features,
                        num_channels, separate_value):
    encoder = RecurrentBackboneEncoder(
        net = MLP(
            input_dim = num_obs_features,
            num_channels = num_channels,
            num_layers = 3,
        ),
        rnn = LSTM(
            in_channels = num_channels,
            hidden_channels = num_channels,
            num_layers = 1,
        ),
    )

    #encoder = BackboneEncoder(
    #    net = MLP(
    #        input_dim = num_obs_features,
    #        num_channels = num_channels,
    #        num_layers = 3,
    #    ),
    #)

    if separate_value:
        backbone = BackboneSeparate(
            process_obs = process_obs,
            actor_encoder = encoder,
            critic_encoder = RecurrentBackboneEncoder(
                net = MLP(
                    input_dim = num_obs_features,
                    num_channels = num_channels,
                    num_layers = 2,
                ),
                rnn = LSTM(
                    in_channels = num_channels,
                    hidden_channels = num_channels,
                    num_layers = 1,
                ),
            )
        )
    else:
        backbone = BackboneShared(
            process_obs = process_obs,
            encoder = encoder,
        )

    return ActorCritic(
        backbone = backbone,
        actor = LinearLayerDiscreteActor(
            [4, 8, 5, 2],
            num_channels,
        ),
        critic = LinearLayerCritic(num_channels),
    )

def setup_tdm_obs(sim):
    self_obs_tensor = sim.self_observation_tensor().to_torch()
    teammate_obs_tensor = sim.teammate_observations_tensor().to_torch()
    opponent_obs_tensor = sim.opponent_observations_tensor().to_torch()
    opponent_masks_tensor = sim.opponent_masks_tensor().to_torch()
    lidar_tensor = sim.lidar_tensor().to_torch()
    hp_tensor = sim.hp_tensor().to_torch()
    magazine_tensor = sim.magazine_tensor().to_torch()
    alive_tensor = sim.alive_tensor().to_torch()

    N, A = self_obs_tensor.shape[0:2]
    batch_size = N * A

    #id_tensor = torch.arange(A).float()
    #if A > 1:
    #    id_tensor = id_tensor / (A - 1)

    #id_tensor = id_tensor.to(device=self_obs_tensor.device)
    #id_tensor = id_tensor.view(1, 2).expand(N, 2).reshape(batch_size, 1)

    self_obs_tensor = self_obs_tensor.view(batch_size, *self_obs_tensor.shape[2:])
    teammate_obs_tensor = teammate_obs_tensor.view(batch_size, *teammate_obs_tensor.shape[2:])
    opponent_obs_tensor = opponent_obs_tensor.view(batch_size, *opponent_obs_tensor.shape[2:])
    opponent_masks_tensor = opponent_masks_tensor.view(batch_size, *opponent_masks_tensor.shape[2:])
    lidar_tensor = lidar_tensor.view(batch_size, *lidar_tensor.shape[2:])
    hp_tensor = hp_tensor.view(batch_size, *hp_tensor.shape[2:])
    magazine_tensor = magazine_tensor.view(batch_size, *magazine_tensor.shape[2:])
    alive_tensor = alive_tensor.view(batch_size, *alive_tensor.shape[2:])

    obs_tensors = [
        self_obs_tensor,
        teammate_obs_tensor,
        opponent_obs_tensor,
        lidar_tensor,
        hp_tensor,
        magazine_tensor,
        alive_tensor,
    ]

    num_obs_features = 0
    for tensor in obs_tensors:
        num_obs_features += math.prod(tensor.shape[1:])

    obs_tensors.append(opponent_masks_tensor)

    num_lidar_samples = lidar_tensor.shape[1]

    return obs_tensors, num_obs_features, num_lidar_samples

class TDMCommon(nn.Module):
    def __init__(self, num_lidar_samples):
        super().__init__()

        self.lidar_normalizer = EMANormalizer(0.99999, [1, num_lidar_samples])
        self.lidar_conv = nn.Conv1d(
            in_channels = 1,
            out_channels = 1,
            kernel_size = num_lidar_samples,
            padding = 'same',
            padding_mode = 'circular',
        )

    def forward(self,
                self_obs,
                teammate_obs,
                opponent_obs,
                lidar,
                hp,
                magazine,
                alive,
                opponent_masks):
        assert_valid_input(self_obs)
        assert_valid_input(teammate_obs)
        assert_valid_input(opponent_obs)
        assert_valid_input(lidar)
        assert_valid_input(hp)
        assert_valid_input(magazine)
        assert_valid_input(alive)
        assert_valid_input(opponent_masks)

        with torch.no_grad():
            lidar_normalized = self.lidar_normalizer(lidar)

        lidar_processed = self.lidar_conv(lidar_normalized.unsqueeze(dim=1))
        lidar_processed = lidar_processed.view(*lidar_normalized.shape)

        return (self_obs, teammate_obs, opponent_obs, lidar_processed, hp,
                magazine, alive, opponent_masks)


class TDMActorNet(nn.Module):
    def __init__(self, num_obs_features, num_channels):
        super().__init__()

        self.mlp = MLP(
            input_dim = num_obs_features,
            num_channels = num_channels,
            num_layers = 2,
        )

        self.normalizer = EMANormalizer(0.99999, [1, num_obs_features])

    def forward(self, obs_tensors):
        self_obs, teammate_obs, opponent_obs, lidar, hp, magazine, alive, opponent_masks = \
            obs_tensors
        
        with torch.no_grad():
            opponent_obs_masked = opponent_obs * opponent_masks

            flattened = torch.cat([
                self_obs.view(self_obs.shape[0], -1),
                teammate_obs.view(teammate_obs.shape[0], -1),
                opponent_obs_masked.view(opponent_obs_masked.shape[0], -1),
                lidar,
                hp.view(hp.shape[0], -1),
                magazine.view(magazine.shape[0], -1),
                alive.view(alive.shape[0], -1),
            ], dim=1)

            normalized = self.normalizer(flattened)

        return self.mlp(normalized)


class TDMCriticNet(nn.Module):
    def __init__(self, num_obs_features, num_channels):
        super().__init__()

        self.mlp = MLP(
            input_dim = num_obs_features,
            num_channels = num_channels,
            num_layers = 2,
        )

        self.normalizer =  EMANormalizer(0.99999, [1, num_obs_features])

    def forward(self, obs_tensors):
        self_obs, teammate_obs, opponent_obs, lidar, hp, magazine, alive, opponent_masks = \
            obs_tensors
        
        with torch.no_grad():
            flattened = torch.cat([
                self_obs.view(self_obs.shape[0], -1),
                teammate_obs.view(teammate_obs.shape[0], -1),
                opponent_obs.view(opponent_obs.shape[0], -1),
                lidar,
                hp.view(opponent_masks.shape[0], -1),
                magazine.view(magazine.shape[0], -1),
                alive.view(alive.shape[0], -1),
            ], dim=1)

            normalized = self.normalizer(flattened)

        return self.mlp(normalized)

def make_tdm_policy(num_obs_features, num_channels, num_lidar_samples):
    obs_common = TDMCommon(num_lidar_samples)

    actor_encoder = RecurrentBackboneEncoder(
        net = TDMActorNet(num_obs_features, num_channels),
        rnn = LSTM(
            in_channels = num_channels,
            hidden_channels = num_channels,
            num_layers = 1,
        ),
    )

    critic_encoder = RecurrentBackboneEncoder(
        net = TDMCriticNet(num_obs_features, num_channels),
        rnn = LSTM(
            in_channels = num_channels,
            hidden_channels = num_channels,
            num_layers = 1,
        ),
    )

    backbone = BackboneSeparate(
        process_obs = obs_common,
        actor_encoder = actor_encoder,
        critic_encoder = critic_encoder,
    )

    return ActorCritic(
        backbone = backbone,
        actor = LinearLayerDiscreteActor(
            [4, 8, 5, 5, 2, 2],
            num_channels,
        ),
        critic = LinearLayerCritic(num_channels),
    )
