import jax
from jax import lax, random, numpy as jnp

import madrona_learn
madrona_learn.cfg_jax_mem(0.7)

from jax.experimental import checkify
import flax
from flax import linen as nn
from flax.core import frozen_dict, FrozenDict
import numpy as np

import argparse
from dataclasses import dataclass
from functools import partial
from time import time, sleep
import os
import json
import pickle
import optax

from madrona_learn import (
    ActionsConfig, ActorCritic, TrainConfig, PPOConfig, PBTConfig,
    ParamExplore, TensorboardWriter, TrainHooks,
)

from madrona_learn.utils import aot_compile, get_checkify_errors
from madrona_learn.train_state import TrainStateManager

import madrona_mp_env
from madrona_mp_env import Task, SimFlags

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--ckpt-dir', type=str, required=True)
arg_parser.add_argument('--tb-dir', type=str, required=True)
arg_parser.add_argument('--in-run-name', type=str, required=True)
arg_parser.add_argument('--out-run-name', type=str, required=True)
arg_parser.add_argument('--start-update', type=int, required=True)
arg_parser.add_argument('--bc-data-dir', type=str, required=True)
arg_parser.add_argument('--kl-data-dir', type=str, required=True)
arg_parser.add_argument('--bf16', action='store_true')
arg_parser.add_argument('--lr', type=float, default=0.01)
arg_parser.add_argument('--num-epochs', type=int, default=100)
arg_parser.add_argument('--minibatch-size', type=int, default=2048)

#arg_parser.add_argument('--game-mode', type=str, required=True)
#arg_parser.add_argument('--scene', type=str, required=True)
#
#arg_parser.add_argument('--randomize-hp-mag', action='store_true')
#arg_parser.add_argument('--use-middle-spawns', action='store_true')
#
#arg_parser.add_argument('--num-worlds', type=int, required=True)
#arg_parser.add_argument('--num-updates', type=int, required=True)
#arg_parser.add_argument('--steps-per-update', type=int, default=50)
#arg_parser.add_argument('--num-bptt-chunks', type=int, default=1)
#arg_parser.add_argument('--num-minibatches', type=int, default=1)
#
#arg_parser.add_argument('--gamma', type=float, default=0.998)
#arg_parser.add_argument('--entropy-loss-coef', type=float, default=0.3)
#arg_parser.add_argument('--value-loss-coef', type=float, default=1.0)
#arg_parser.add_argument('--clip-value-loss', action='store_true')
#arg_parser.add_argument('--pbt-ensemble-size', type=int, default=1)
#arg_parser.add_argument('--pbt-past-policies', type=int, default=0)
#
#arg_parser.add_argument('--num-channels', type=int, default=1024)
#arg_parser.add_argument('--separate-value', action='store_true')
#arg_parser.add_argument('--fp16', action='store_true')
#
#arg_parser.add_argument('--gpu-sim', action='store_true')
#arg_parser.add_argument('--profile-port', type=int, default=None)
#
#arg_parser.add_argument('--static-past', action='store_true')
#arg_parser.add_argument('--static-flip-teams', action='store_true')
#
#arg_parser.add_argument('--full-team-policy', action='store_true')
#arg_parser.add_argument('--eval-frequency', type=int, default=500)
#
#arg_parser.add_argument('--curriculum-data', type=str)

args = arg_parser.parse_args()

from jax_policy import make_policy

tb_writer = TensorboardWriter(os.path.join(args.tb_dir, args.out_run_name))

dev = jax.devices()[0]

with open(f"{args.bc_data_dir}/shapes", 'r') as f:
    metadata = json.loads(f.read())

print(metadata)

if args.bf16:
    dtype = jnp.bfloat16
else:
    dtype = jnp.float32

actions_cfg = ActionsConfig(
    actions_num_buckets = [ 4, 8, 5, 5, 2, 2, 3 ],
)

policy = make_policy(dtype, actions_cfg)

restore_ckpt = os.path.realpath(os.path.join(
    os.path.join(args.ckpt_dir, args.in_run_name), str(args.start_update)))

def load_data(data_dir):
    with open(f"{data_dir}/rewards", 'rb') as f:
        rewards_data = np.fromfile(f, dtype=np.float32)
        rewards_data = rewards_data.reshape(-1, *metadata['rewards'])

    with open(f"{data_dir}/actions", 'rb') as f:
        actions_data = np.fromfile(f, dtype=np.int32)
        actions_data = actions_data.reshape(-1, *metadata['actions'])

    with open(f"{data_dir}/actions_logits", 'rb') as f:
        actions_logits_data = np.fromfile(f, dtype=np.float32)
        actions_logits_data = actions_logits_data.reshape(-1, *metadata['actions_logits'])

    with open(f"{data_dir}/rnn_starts", 'rb') as f:
        rnn_starts_data = np.fromfile(f, dtype=jnp.bfloat16)
        rnn_starts_data = rnn_starts_data.reshape(-1, *metadata['rnn_states'])

    obs_data = {}

    for k, shape in metadata['obs'].items():
        dtype = np.float32
        if k == 'magazine':
            dtype = np.int32

        with open(f"{data_dir}/{k}", 'rb') as f:
            obs_data[k] = np.fromfile(f, dtype=dtype)
            obs_data[k] = obs_data[k].reshape(-1, *shape)

    obs_data = frozen_dict.freeze(obs_data)

    return {
        'obs': jax.tree.map(lambda x: jax.device_put(x), obs_data),
        'rewards': jax.device_put(rewards_data),
        'actions': jax.device_put(actions_data),
        'actions_logits': jax.device_put(actions_logits_data),
        'rnn_starts': jax.device_put(rnn_starts_data),
    }

bc_train_data = load_data(args.bc_data_dir)
#kl_train_data = load_data(args.kl_data_dir)
kl_train_data = bc_train_data

def convert_rnn_states(states):
    x = (
        ([states[:, 0, 0, 0, ...]], [states[:, 0, 1, 0, ...]]),
        ([states[:, 1, 0, 0, ...]], [states[:, 1, 1, 0, ...]]),
    )

    return x

def load_ckpt():
    with open('/tmp/t_cfg', 'rb') as f:
        cfg = pickle.load(f)

    example_obs = jax.tree.map(lambda x: x[0:cfg.pbt.num_train_policies, 0, 0],
                               bc_train_data['obs'])
    example_obs = example_obs.copy({
        'agent_map': jnp.zeros((cfg.pbt.num_train_policies, 16, 16, 4))})
    example_obs = example_obs.copy({
        'unmasked_agent_map': jnp.zeros((cfg.pbt.num_train_policies, 16, 16, 4))})

    example_rnn_states = convert_rnn_states(
        bc_train_data['rnn_starts'][0:cfg.pbt.num_train_policies, 0])

    train_state_mgr = TrainStateManager.create(
        policy = policy, 
        cfg = cfg,
        algo = cfg.algo.setup(),
        init_user_state_cb = lambda *args: None,
        base_rng = random.key(5),
        example_obs = example_obs,
        example_rnn_states = example_rnn_states,
        use_competitive_mmr = True,
        checkify_errors = get_checkify_errors(),
    )
    
    train_state_mgr, ppo_next_update = train_state_mgr.load(restore_ckpt)

    return train_state_mgr, ppo_next_update

train_state_mgr, ppo_next_update = load_ckpt()

def iter(train_state_mgr,
         bc_mb_obs,
         bc_mb_rewards,
         bc_mb_actions,
         bc_mb_rnn_starts):
    @jax.vmap
    def vmap_update(policy_state, train_state):
        def fwd_pass(params):
            fake_dones = jnp.zeros(bc_mb_rewards.shape, dtype=jnp.int32)

            def reorder_input(x):
                x = x.swapaxes(0, 1)
                x = x.reshape(x.shape[0], -1, *x.shape[3:])
                return x

            processed_obs = bc_mb_obs.copy({
                'agent_map': jnp.zeros((*bc_mb_rewards.shape[:-1], 16, 16, 4))})
            processed_obs = processed_obs.copy({
                'unmasked_agent_map': jnp.zeros((*bc_mb_rewards.shape[:-1], 16, 16, 4))})

            processed_obs = jax.tree.map(reorder_input, processed_obs)
            processed_obs = policy_state.obs_preprocess.preprocess(
                policy_state.obs_preprocess_state, processed_obs, False)

            return policy_state.apply_fn(
                { 'params': params, 'batch_stats': policy_state.batch_stats },
                convert_rnn_states(
                    bc_mb_rnn_starts.reshape(-1, *bc_mb_rnn_starts.shape[2:])),
                reorder_input(fake_dones),
                reorder_input(bc_mb_actions),
                processed_obs,
                train=True,
                method='update',
                mutable=['batch_stats'],
            )

        def bc_loss_fn(params):
            fwd_results, _ = fwd_pass(params)

            return -jnp.mean(fwd_results['log_probs'])

        def loss_fn(params):
            return bc_loss_fn(params), ()

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(policy_state.params)

        loss = aux[0]
        jax.debug.print("{}", loss)

        with jax.numpy_dtype_promotion('standard'):
            param_updates, new_opt_state = train_state.tx.update(
                grads, train_state.opt_state, policy_state.params)
        new_params = optax.apply_updates(
            policy_state.params, param_updates)

        policy_state = policy_state.update(
            params = new_params,
        )

        train_state = train_state.update(
            opt_state = new_opt_state
        )

        return policy_state, train_state


    num_train_policies = train_state_mgr.train_states.update_prng_key.shape[0]
    
    train_policy_states = jax.tree.map(
        lambda x: x[0:num_train_policies],
        train_state_mgr.policy_states)

    train_policy_states, train_states = vmap_update(
        train_policy_states,
        train_state_mgr.train_states)

    policy_states = jax.tree.map(
        lambda full, new: full.at[0:num_train_policies].set(new),
        train_state_mgr.policy_states, train_policy_states)

    return train_state_mgr.replace(
        policy_states = policy_states,
        train_states = train_states,
    )

@partial(jax.jit, donate_argnums=[0])
def epoch(train_state_mgr, bc_train_data, kl_train_data):
    rnd = train_state_mgr.pbt_rng
    epoch_rnd, next_rnd = random.split(rnd)
    train_state_mgr = train_state_mgr.replace(pbt_rng = next_rnd)

    load_idx_perm = random.permutation(
        epoch_rnd, bc_train_data['rewards'].shape[0])

    remainder = load_idx_perm.shape[0] % args.minibatch_size
    if remainder > 0:
        load_idx_perm = jnp.concatenate(
            [load_idx_perm, load_idx_perm[:args.minibatch_size - remainder]], axis=0)

    load_idx_perm = load_idx_perm.reshape(-1, args.minibatch_size) 

    print(load_idx_perm.shape)

    def minibatch(minibatch_idx, carry):
        train_state_mgr = carry

        load_indices = load_idx_perm[minibatch_idx]

        mb_obs = jax.tree.map(
            lambda x: x[load_indices], bc_train_data['obs'])
        mb_rewards = bc_train_data['rewards'][load_indices]
        mb_actions = bc_train_data['actions'][load_indices]

        mb_rnn_starts = bc_train_data['rnn_starts'][load_indices]

        return iter(train_state_mgr, mb_obs, mb_rewards,
                    mb_actions, mb_rnn_starts)

    train_state_mgr = lax.fori_loop(0, load_idx_perm.shape[0], minibatch, 
                                    train_state_mgr)

    return train_state_mgr

def train(train_state_mgr):
    for i in range(args.num_epochs):
        print(i)
        train_state_mgr = epoch(train_state_mgr, bc_train_data, kl_train_data)

    train_state_mgr.save(
        ppo_next_update,
        os.path.realpath(f"{args.ckpt_dir}/{args.out_run_name}/{ppo_next_update}"))

if __name__ == "__main__":
    try:
        train(train_state_mgr)
    except:
        tb_writer.flush()
        raise
    
    tb_writer.flush()
