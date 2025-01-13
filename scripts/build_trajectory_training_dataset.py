#!/usr/bin/env python

import sys
import sqlite3
import numpy as np
import pickle
import json
from jax import numpy as jnp

num_worlds = sys.argv[1]
sql_db_path = sys.argv[2]
trajectory_db_path = sys.argv[3]

con = sqlite3.connect(sql_db_path)
db = con.cursor()

with open(trajectory_db_path, 'rb') as f:
    trajectories = np.fromfile(f, dtype=np.int64)

data_dir = sys.argv[4]
bc_out_dir = sys.argv[5]
kl_out_dir = sys.argv[6]

num_agents = 12

obs_shapes = {
    'alive': (1,),
    'filters_state': (1,),
    'fwd_lidar': (2, 32, 4),
    'hp': (1,),
    'magazine': (2,),
    'opponent_last_known_positions': (6, 3),
    'opponent_masks': (6, 1),
    'opponent_positions': (6, 3),
    'opponents': (6, 30),
    'opponents_last_known': (6, 30),
    'rear_lidar': (2, 8, 4),
    'self': (51,),
    'self_pos': (3,),
    'teammate_positions': (5, 3),
    'teammates': (5, 39),
}

actions_shape = (7,)
actions_logits_shape = (29,)

with open(f"{data_dir}/rewards", 'rb') as f:
    rewards = np.fromfile(f, dtype=np.float32)
    rewards = rewards.reshape(-1, num_agents, 1)
    print(rewards.shape)

with open(f"{data_dir}/actions", 'rb') as f:
    actions = np.fromfile(f, dtype=np.int32)
    actions = actions.reshape(-1, num_agents, *actions_shape)
    print(actions.shape)

with open(f"{data_dir}/action_logits", 'rb') as f:
    actions_logits = np.fromfile(f, dtype=np.float32)
    actions_logits = actions_logits.reshape(-1, num_agents, *actions_logits_shape)

obs_files = {}

for k in obs_shapes.keys():
    obs_files[k] = open(f"{data_dir}/{k}", 'rb')

rnn_states_file = open(f"{data_dir}/rnn_states", 'rb')
rnn_states_shape = (12, 2, 2, 1, 512)

seq_len = 20

trajectories = trajectories.reshape(-1, seq_len, 2)

#rewards: (1920, 1)

def load_players_for_step(step_id):
    res = db.execute(f"""
        SELECT
            ps.pos_x, ps.pos_y, ps.pos_z, ps.yaw, ps.pitch,
            ps.num_bullets, ps.is_reloading, ps.fired_shot,
            ps.hp, ps.stand_state
        FROM player_states AS ps
        WHERE ps.step_id = {step_id}
    """)

    rows = res.fetchall()

    return rows

def load_match_state_for_step(step_id):
    res = db.execute(f"""
      SELECT ms.step_idx, ms.cur_zone, ms.cur_zone_controller,
             ms.zone_steps_remaining, ms.zone_steps_until_point
      FROM match_steps AS ms
      WHERE ms.id = {step_id}
    """)

    return res.fetchone()

def get_ordered_steps():
    res = db.execute(f"""
        SELECT id FROM match_steps ORDER BY match_id, step_idx
    """)

    return res.fetchall()

def compute_non_matching():
    trajectory_step_ids = trajectories[..., 0].ravel()
    print(len(trajectory_step_ids))
    flattened_trajectories = trajectory_step_ids
    full_range = np.asarray(get_ordered_steps()).ravel()
    print(len(full_range))

    in_array = np.in1d(full_range, flattened_trajectories)
    missing_mask = ~in_array

    missing_vals = full_range[missing_mask]

    return missing_vals

non_matching = compute_non_matching()
print(len(non_matching))

def dump_trajectories(out_dir, trajectories):
    out_actions_file = open(f"{out_dir}/actions", 'wb')
    out_actions_logits_file = open(f"{out_dir}/actions_logits", 'wb')
    out_rewards_file = open(f"{out_dir}/rewards", 'wb')
    out_rnn_states_file = open(f"{out_dir}/rnn_starts", 'wb')
    
    out_obs_files = {}
    
    for k in obs_files.keys():
        out_obs_files[k] = open(f"{out_dir}/{k}", 'wb')

    def dump_trajectory(steps):

        trajectory_actions = []
        trajectory_actions_logits = []
        trajectory_rewards = []
        trajectory_obs = {}
    
        for k in obs_files.keys():
            trajectory_obs[k] = []
    
        rnn_start_idx = steps[0, 0] - 1
        team_idx = steps[0, 1]
        if team_idx == 0:
            team_slice = [0, 6]
        else:
            team_slice = [6, 12]
    
        if trajectory_idx % 100 == 0:
            print(trajectory_idx, rnn_start_idx)
        
        rnn_state_num_floats = np.prod(rnn_states_shape)
    
        rnn_states_file.seek(0)
        rnn_state_data = np.fromfile(
            rnn_states_file, count=rnn_state_num_floats,
            dtype=jnp.bfloat16, offset = 2 * rnn_state_num_floats * rnn_start_idx)
        rnn_state_data = rnn_state_data.reshape(rnn_states_shape)
        rnn_state_data = rnn_state_data[team_slice[0]:team_slice[1]]
    
        rnn_state_data.tofile(out_rnn_states_file)
    
        for trajectory_offset in range(steps.shape[0]):
            step_id = steps[trajectory_offset, 0]
            team_id = steps[trajectory_offset, 1]
            global_idx = step_id - 1
    
            step_actions = actions[global_idx]
            step_rewards = rewards[global_idx]
            step_actions_logits = actions_logits[global_idx]

            step_actions = step_actions[team_slice[0]:team_slice[1]]
            step_actions_logits = step_actions_logits[team_slice[0]:team_slice[1]]
            step_rewards = step_rewards[team_slice[0]:team_slice[1]]

            trajectory_actions.append(step_actions)
            trajectory_actions_logits.append(step_actions_logits)
            trajectory_rewards.append(step_rewards)
    
            for k in obs_files.keys():
                shape = obs_shapes[k]
                file = obs_files[k]
    
                file.seek(0)
    
                shape = (num_agents, *shape)
                num_floats = np.prod(shape)
    
                dtype = np.float32
                if k == 'magazine':
                    dtype = np.int32
    
                data = np.fromfile(file, count=num_floats,
                                   offset = 4 * num_floats * global_idx,
                                   dtype=dtype)
    
                data = data.reshape(*shape)

                data = data[team_slice[0]:team_slice[1]]
    
                trajectory_obs[k].append(data)
    
        seq_actions = np.stack(trajectory_actions)
        seq_actions_logits = np.stack(trajectory_actions_logits)
        seq_rewards = np.stack(trajectory_rewards)
    
        seq_actions.tofile(out_actions_file)
        seq_actions_logits.tofile(out_actions_logits_file)
        seq_rewards.tofile(out_rewards_file)
    
        seq_obs = {}
        for k, v in trajectory_obs.items():
            seq_obs[k] = np.stack(v)
            seq_obs[k].tofile(out_obs_files[k])

        return seq_actions, seq_actions_logits, seq_rewards, seq_obs, rnn_state_data
    
    for trajectory_idx in range(trajectories.shape[0]):
        seq_actions, seq_actions_logits, seq_rewards, seq_obs, rnn_state_data = dump_trajectory(
            trajectories[trajectory_idx])
    
    with open(f"{out_dir}/shapes", "w") as f:
        metadata = {
            'actions': seq_actions.shape,
            'actions_logits': seq_actions_logits.shape,
            'rewards': seq_rewards.shape,
            'rnn_states': rnn_state_data.shape,
        }
    
        metadata['obs'] = {}
        for k, v in seq_obs.items():
            metadata['obs'][k] = v.shape
    
        f.write(json.dumps(metadata, indent=2))

print(trajectories.shape)

dump_trajectories(bc_out_dir, trajectories)

truncated = len(non_matching) // seq_len * seq_len
non_matching = non_matching[:truncated]
non_matching = non_matching.reshape(-1, seq_len)

non_matching = non_matching[np.random.permutation(
    non_matching.shape[0])[:10 * trajectories.shape[0]]]

non_matching = np.stack([non_matching, np.random.randint(low=0, high=2, size=non_matching.shape, dtype=np.int64)], axis=-1)
print(non_matching.shape)

dump_trajectories(kl_out_dir, non_matching)
