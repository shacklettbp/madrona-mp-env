import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn

import argparse
from functools import partial
import numpy as np

import madrona_mp_env
from madrona_mp_env import Task, SimFlags

import madrona_learn
from madrona_learn import (
    ActorCritic,
    ActionsConfig,
    EvalConfig,
    eval_load_ckpt,
    eval_policies,
)

from common import print_elos

madrona_learn.cfg_jax_mem(0.6)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, default=200)
arg_parser.add_argument('--num-policies', type=int, default=1)
arg_parser.add_argument('--single-policy', type=int, default=None)
arg_parser.add_argument('--crossplay', action='store_true')
arg_parser.add_argument('--crossplay-include-past', action='store_true')

arg_parser.add_argument('--game-mode', type=str, required=True)
arg_parser.add_argument('--scene', type=str, required=True)

arg_parser.add_argument('--num-channels', type=int, default=256)

arg_parser.add_argument('--ckpt-path', type=str, required=True)
arg_parser.add_argument('--record', type=str, default=None)
arg_parser.add_argument('--event-log', type=str, default=None)

arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--fp16', action='store_true')
arg_parser.add_argument('--bf16', action='store_true')

arg_parser.add_argument('--full-team-policy', action='store_true')

args = arg_parser.parse_args()

if args.full_team_policy:
    from jax_full_team_policy import make_policy 
else:
    from jax_policy import make_policy 

team_size = 6

dev = jax.devices()[0]

if args.fp16:
    dtype = jnp.float16
elif args.bf16:
    dtype = jnp.bfloat16
else:
    dtype= jnp.float32

actions_cfg = ActionsConfig(
    actions_num_buckets = [ 4, 8, 5, 5, 2, 2, 3 ],
)

policy = make_policy(dtype, args.scene, actions_cfg)

game_mode = getattr(Task, args.game_mode)

if args.single_policy != None:
    assert not args.crossplay
    policy_states, num_policies = eval_load_ckpt(
        policy, args.ckpt_path, single_policy = args.single_policy)
elif args.crossplay:
    policy_states, num_policies = eval_load_ckpt(
        policy, args.ckpt_path,
        train_only=False if args.crossplay_include_past else True)

print(num_policies)

sim_flags = SimFlags.Default

if args.full_team_policy:
    sim_flags |= SimFlags.FullTeamPolicy

if game_mode == Task.ZoneCaptureDefend:
    sim_flags |= SimFlags.HardcodedSpawns
    #sim_flags |= SimFlags.StaticFlipTeams

sim = madrona_mp_env.SimManager(
    exec_mode = madrona_mp_env.madrona.ExecMode.CUDA if args.gpu_sim else madrona_mp_env.madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    auto_reset = True,
    sim_flags = sim_flags, #SimFlags.EnableCurriculum,
    #sim_flags = SimFlags.EnableCurriculum,
    #sim_flags = SimFlags.RandomFlipTeams | SimFlags.StaggerStarts,
    #sim_flags = SimFlags.Default,
    task_type = game_mode,
    team_size = team_size,
    num_pbt_policies = num_policies,
    rand_seed = 10,
    policy_history_size = 1,
    scene_path = args.scene,
    record_log_path = args.record,
    event_log_path = args.event_log,
)

jax_gpu = jax.devices()[0].platform == 'gpu'

sim_fns = sim.jax(jax_gpu)

total_num_swaps = np.zeros((4,), np.int32)

def host_cb(obs, actions, action_probs, values, dones, rewards, num_zone_swaps):
    global total_num_swaps

    total_num_swaps += num_zone_swaps

    if num_zone_swaps.sum() > 0:
        print(total_num_swaps)

    #print(obs)

    #print("Actions:", actions)

    #print("Move Amount Probs")
    #print(" ", np.array_str(action_probs[0][0], precision=2, suppress_small=True))
    #print(" ", np.array_str(action_probs[0][1], precision=2, suppress_small=True))

    #print("Move Angle Probs")
    #print(" ", np.array_str(action_probs[1][0], precision=2, suppress_small=True))
    #print(" ", np.array_str(action_probs[1][1], precision=2, suppress_small=True))

    #print("Yaw Rotate Probs")
    #print(" ", np.array_str(action_probs[2][0], precision=2, suppress_small=True))
    #print(" ", np.array_str(action_probs[2][1], precision=2, suppress_small=True))

    #print("Pitch Rotate Probs")
    #print(" ", np.array_str(action_probs[3][0], precision=2, suppress_small=True))
    #print(" ", np.array_str(action_probs[3][1], precision=2, suppress_small=True))

    #print("Fire Probs")
    #print(" ", np.array_str(action_probs[4][0], precision=2, suppress_small=True))
    #print(" ", np.array_str(action_probs[4][1], precision=2, suppress_small=True))

    #print("Reload Probs")
    #print(" ", np.array_str(action_probs[5][0], precision=2, suppress_small=True))
    #print(" ", np.array_str(action_probs[5][1], precision=2, suppress_small=True))

    #print("Rewards:", rewards)

    return ()

step_idx = 0

def print_step_cb():
    global step_idx
    print("Step:", step_idx)
    step_idx += 1

def iter_cb(step_data):
    if args.record == None:
        cb = partial(jax.experimental.io_callback, host_cb, ())

        dones = step_data['dones'].reshape(args.num_worlds, -1)[:, 0, None, None]

        # episode_results format
        # 0: winResult
        # 1-2: teamTotalKills
        # 3-4: teamObjectivePoints
        # 5 - : zoneStats

        zone_stats = step_data['episode_results'][:, 5:]
        zone_stats = zone_stats.reshape(-1, 4, 5)

        zone_stats = jnp.where(dones, zone_stats, jnp.zeros((1, 4, 5), dtype=jnp.int32))

        step_num_swaps = zone_stats[:, :, 0].sum(axis=0)

        cb(step_data['obs'],
           step_data['actions'],
           step_data['action_probs'],
           step_data['values'],
           dones,
           step_data['rewards'],
           step_num_swaps)
    else:
        cb = partial(jax.experimental.io_callback, print_step_cb, ())
        cb()

if args.full_team_policy:
    eval_team_size = 1
else:
    eval_team_size = team_size

eval_cfg = EvalConfig(
    num_worlds = args.num_worlds,
    num_teams = 2,
    team_size = eval_team_size,
    actions = actions_cfg,
    num_eval_steps = args.num_steps,
    policy_dtype = dtype,
    eval_competitive = True,
    use_deterministic_policy = False,
    reward_gamma = 0.998,
    #custom_policy_ids = [ -1 ],
)

print_elos(policy_states.mmr.elo)

mmrs = eval_policies(dev, eval_cfg, sim_fns,
    policy, jnp.array([0, 0], jnp.int32), policy_states, iter_cb)

print_elos(mmrs.elo)

del sim
