import jax
from jax import lax, random, numpy as jnp

import madrona_learn
madrona_learn.cfg_jax_mem(0.7)

from jax.experimental import checkify
import flax
from flax import linen as nn
import numpy as np

import argparse
from dataclasses import dataclass
from functools import partial
from time import time, sleep
import os

from madrona_learn import (
    ActorCritic, TrainConfig, PPOConfig, PBTConfig,
    ParamExplore, TensorboardWriter, TrainHooks,
)

import madrona_mp_env
from madrona_mp_env import Task, SimFlags

import pickle

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--ckpt-dir', type=str, required=True)
arg_parser.add_argument('--tb-dir', type=str, required=True)
arg_parser.add_argument('--run-name', type=str, required=True)
arg_parser.add_argument('--restore', type=int)
arg_parser.add_argument('--game-mode', type=str, required=True)
arg_parser.add_argument('--scene', type=str, required=True)

arg_parser.add_argument('--randomize-hp-mag', action='store_true')
arg_parser.add_argument('--use-middle-spawns', action='store_true')

arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-updates', type=int, required=True)
arg_parser.add_argument('--steps-per-update', type=int, default=50)
arg_parser.add_argument('--num-bptt-chunks', type=int, default=1)
arg_parser.add_argument('--num-minibatches', type=int, default=1)

arg_parser.add_argument('--lr', type=float, default=0.01)
arg_parser.add_argument('--gamma', type=float, default=0.998)
arg_parser.add_argument('--entropy-loss-coef', type=float, default=0.3)
arg_parser.add_argument('--value-loss-coef', type=float, default=1.0)
arg_parser.add_argument('--clip-value-loss', action='store_true')
arg_parser.add_argument('--pbt-ensemble-size', type=int, default=1)
arg_parser.add_argument('--pbt-past-policies', type=int, default=0)

arg_parser.add_argument('--num-channels', type=int, default=1024)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')
arg_parser.add_argument('--bf16', action='store_true')

arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--profile-port', type=int, default=None)

arg_parser.add_argument('--static-past', action='store_true')
arg_parser.add_argument('--static-flip-teams', action='store_true')

arg_parser.add_argument('--full-team-policy', action='store_true')
arg_parser.add_argument('--eval-frequency', type=int, default=500)

arg_parser.add_argument('--curriculum-data', type=str)

args = arg_parser.parse_args()

if args.full_team_policy:
    from jax_full_team_policy import make_policy
else:
    from jax_policy import make_policy, actions_config

game_mode = getattr(Task, args.game_mode)


sim_flags = SimFlags.Default

if args.full_team_policy:
    sim_flags |= SimFlags.FullTeamPolicy

if args.randomize_hp_mag:
    sim_flags |= SimFlags.RandomizeHPMagazine

if args.use_middle_spawns:
    sim_flags |= SimFlags.SpawnInMiddle

sim_flags |= SimFlags.StaggerStarts 

if game_mode == Task.ZoneCaptureDefend:
    sim_flags |= SimFlags.HardcodedSpawns
    #sim_flags |= SimFlags.StaticFlipTeams
    sim_flags |= SimFlags.RandomFlipTeams
else:
    sim_flags |= SimFlags.RandomFlipTeams

if args.static_flip_teams:
    sim_flags |= SimFlags.StaticFlipTeams

#sim_flags |= SimFlags.EnableCurriculum

team_size = 6

tb_writer = TensorboardWriter(os.path.join(args.tb_dir, args.run_name))

sim = madrona_mp_env.SimManager(
    exec_mode = madrona_mp_env.madrona.ExecMode.CUDA if args.gpu_sim else madrona_mp_env.madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    auto_reset = True,
    sim_flags = sim_flags,
    task_type = game_mode,
    team_size = team_size,
    rand_seed = 5,
    num_pbt_policies = args.pbt_ensemble_size + args.pbt_past_policies,
    policy_history_size = args.pbt_past_policies,
    scene_path = args.scene,
    curriculum_data_path = args.curriculum_data,
)

jax_gpu = jax.devices()[0].platform == 'gpu'

sim_fns = sim.jax(jax_gpu)

dev = jax.devices()[0]

if args.full_team_policy:
    if game_mode == Task.Turret:
        num_agents_per_world = 1
        train_team_size = 1
    else:
        num_agents_per_world = 2
        train_team_size = 1
else:
    if game_mode == Task.Turret:
        num_agents_per_world = team_size
        train_team_size = team_size
    else:
        num_agents_per_world = team_size * 2
        train_team_size = team_size

if args.pbt_ensemble_size != 1 or args.pbt_past_policies != 0:
    pbt_cfg = PBTConfig(
        num_teams = 2,
        team_size = train_team_size,
        num_train_policies = args.pbt_ensemble_size,
        num_past_policies = args.pbt_past_policies,
        self_play_portion = 0.75 if not args.static_past else 0,
        cross_play_portion = 0.125 if not args.static_past else 0,
        past_play_portion = 0.125 if not args.static_past else 1,
        #self_play_portion = 0.125 if not args.static_past else 0,
        #cross_play_portion = 0.5 if not args.static_past else 0,
        #past_play_portion = 0.375 if not args.static_past else 1,
        #reward_hyper_params_explore = {
        #    'team_spirit': ParamExplore(
        #        base = 1.0,
        #        min_scale = 0.0,
        #        max_scale = 1.0,
        #        clip_perturb = True,
        #    ),
        #    'shot_scale': ParamExplore(
        #        base = 0.05,
        #        min_scale = 0.1,
        #        max_scale = 10,
        #        log10_scale = True,
        #    ),
        #    'explore_scale': ParamExplore(
        #        base = 0.001,
        #        min_scale = 0.1,
        #        max_scale = 10,
        #        log10_scale = True,
        #    ),
        #    'in_zone_scale': ParamExplore(
        #        base = 0.05,
        #        min_scale = 0.1,
        #        max_scale = 10,
        #        log10_scale = True,
        #    ),
        #    'zone_team_context_scale': ParamExplore(
        #        base = 0.01,
        #        min_scale = 0.1,
        #        max_scale = 10,
        #        log10_scale = True,
        #    ),
        #    'zone_team_ctrl_scale': ParamExplore(
        #        base = 0.1,
        #        min_scale = 0.1,
        #        max_scale = 10,
        #        log10_scale = True,
        #    ),
        #    'zone_team_dist_scale': ParamExplore(
        #        base = 0.005,
        #        min_scale = 0.1,
        #        max_scale = 10,
        #        log10_scale = True,
        #    ),
        #    'zone_earned_point_scale': ParamExplore(
        #        base = 1.0,
        #        min_scale = 0.5,
        #        max_scale = 5,
        #        log10_scale = True,
        #    ),
        #    'breadcrumb_scale': ParamExplore(
        #        base = 0.1,
        #        min_scale = 0.1,
        #        max_scale = 10,
        #        log10_scale = True,
        #    ),
        #}
    )
else:
    pbt_cfg = None

if args.fp16:
    dtype = jnp.float16
elif args.bf16:
    dtype = jnp.bfloat16
else:
    dtype = jnp.float32


if pbt_cfg:
    lr = ParamExplore(
        base = args.lr,
        min_scale = 0.1,
        max_scale = 10.0,
        log10_scale = True,
    )

    entropy_coef = ParamExplore(
        base = args.entropy_loss_coef,
        min_scale = 0.1,
        max_scale = 10.0,
        log10_scale = True,
    )
else:
    lr = args.lr
    entropy_coef = args.entropy_loss_coef

cfg = TrainConfig(
    num_worlds = args.num_worlds,
    num_agents_per_world = num_agents_per_world,
    num_updates = args.num_updates,
    actions = actions_config,
    steps_per_update = args.steps_per_update,
    num_bptt_chunks = args.num_bptt_chunks,
    lr = lr,
    gamma = args.gamma,
    gae_lambda = 0.95,
    algo = PPOConfig(
        num_epochs = 2,
        num_mini_batches = args.num_minibatches,
        clip_coef = 0.2,
        #value_loss_coef = args.value_loss_coef,
        value_loss_coef = 0.5,
        entropy_coef = entropy_coef,
        max_grad_norm = 0.5,
        clip_value_loss = args.clip_value_loss,
        huber_value_loss = True,
    ),
    pbt = pbt_cfg,
    dreamer_v3_critic = False,
    normalize_values = True,
    value_normalizer_decay = 0.999,
    compute_dtype = dtype,
    seed = 5,
    metrics_buffer_size = 10,
    #baseline_policy_id = -1,
    #custom_policy_ids = [-1],
)

with open("/tmp/t_cfg", 'wb') as f:
    pickle.dump(cfg, f)

policy = make_policy(dtype)

restore_ckpt = None
if args.restore:
    restore_ckpt = os.path.join(
        os.path.join(args.ckpt_dir, args.run_name), str(args.restore))

last_time = 0
last_update = 0

def _log_metrics_host_cb(training_mgr):
    global last_time, last_update

    update_id = int(training_mgr.update_idx)

    cur_time = time()
    update_diff = update_id - last_update

    print(f"Update: {update_id}")
    if last_time != 0:
        print("  FPS:", args.num_worlds * args.steps_per_update * update_diff / (cur_time - last_time))

    last_time = cur_time
    last_update = update_id

    #metrics.pretty_print()

    if args.pbt_ensemble_size > 0:
        old_printopts = np.get_printoptions()
        np.set_printoptions(formatter={'float_kind':'{:.1e}'.format}, linewidth=150)

        lrs = np.asarray(training_mgr.state.train_states.hyper_params.lr)
        entropy_coefs = np.asarray(
            training_mgr.state.train_states.hyper_params.entropy_coef)

        np.set_printoptions(**old_printopts)

        print()

        elos = training_mgr.state.policy_states.mmr.elo

        for i in range(elos.shape[0]):
            tb_writer.scalar(f"p{i}/elo", elos[i], update_id)

        num_train_policies = lrs.shape[0]
        for i in range(lrs.shape[0]):
            tb_writer.scalar(f"p{i}/lr", lrs[i], update_id)
            tb_writer.scalar(f"p{i}/entropy_coef", entropy_coefs[i], update_id)

    training_mgr.log_metrics_tensorboard(tb_writer)

    return ()


def update_loop(training_mgr):
    assert args.eval_frequency % cfg.metrics_buffer_size == 0

    def inner_iter(i, training_mgr):
        return training_mgr.update_iter()

    def outer_iter(i, training_mgr):
        training_mgr = lax.fori_loop(
            0, cfg.metrics_buffer_size, inner_iter, training_mgr)

        jax.experimental.io_callback(
            _log_metrics_host_cb, (), training_mgr, ordered=True)

        return training_mgr

    return lax.fori_loop(0, args.eval_frequency // cfg.metrics_buffer_size,
                         outer_iter, training_mgr)

def update_population(training_mgr):
    training_mgr, elo_deltas = madrona_learn.eval_elo(
        training_mgr, 3600,
        eval_sim_ctrl=jnp.array([1, 0, 0], jnp.int32),
        train_sim_ctrl=jnp.array([0, 1, 1], jnp.int32))

    training_mgr = madrona_learn.update_population(training_mgr, elo_deltas)

    return training_mgr

def train():
    global last_time 

    training_mgr = madrona_learn.init_training(dev, cfg, sim_fns, policy,
        init_sim_ctrl=jnp.array([0, 1, 1], jnp.int32),
        restore_ckpt=restore_ckpt,
        profile_port=args.profile_port)

    assert training_mgr.update_idx % args.eval_frequency == 0
    num_outer_iters = ((args.num_updates - int(training_mgr.update_idx)) //
        args.eval_frequency)

    update_loop_compiled = madrona_learn.aot_compile(update_loop, training_mgr)

    update_population_compiled = madrona_learn.aot_compile(update_population, training_mgr)

    last_time = time()

    for i in range(num_outer_iters):
        err, training_mgr = update_loop_compiled(training_mgr)
        err.throw()

        err, training_mgr = update_population_compiled(training_mgr)
        err.throw()

        print(training_mgr.state.policy_states.mmr.elo)

        training_mgr.save_ckpt(f"{args.ckpt_dir}/{args.run_name}")
    
    madrona_learn.stop_training(training_mgr)

if __name__ == "__main__":
    try:
        train()
    except:
        tb_writer.flush()
        raise
    
    tb_writer.flush()
    del sim
