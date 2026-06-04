# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the SAPG algorithm through the OnPolicyRunner."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

NUM_ENVS = 8
OBS_DIM = 8
NUM_ACTIONS = 4
MAX_EP_LEN = 7  # short so trajectories reset within a rollout (exercises recurrent split)


class DummyEnv(VecEnv):
    """Minimal VecEnv that returns random observations and rewards."""

    def __init__(self, device: str = "cpu") -> None:
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = MAX_EP_LEN
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}

    def get_observations(self) -> TensorDict:
        data = {"policy": torch.randn(self.num_envs, OBS_DIM, device=self.device)}
        return TensorDict(data, batch_size=[self.num_envs], device=self.device)

    def step(self, actions: torch.Tensor):
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).float()
        self.episode_length_buf[dones.bool()] = 0
        obs = self.get_observations()
        rewards = torch.randn(self.num_envs, device=self.device)
        extras = {"time_outs": torch.zeros(self.num_envs, device=self.device)}
        return obs, rewards, dones, extras


def _make_cfg(model_type: str, aggregation: str) -> dict:
    cfg: dict = {
        "num_steps_per_env": 8,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "SAPG",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "num_blocks": 2,
            "aggregation": aggregation,
            "off_policy_ratio": 1,
            "off_policy_loss_coef": 1.0,
            "latent_dim": 8,
            "entropy_coef_range": 0.01,
        },
    }
    if model_type == "rnn":
        cfg["actor"] = {
            "class_name": "SAPGModel",
            "model_type": "rnn",
            "hidden_dims": [32],
            "rnn_type": "lstm",
            "rnn_hidden_dim": 16,
            "rnn_num_layers": 1,
            "distribution_cfg": {"class_name": "GaussianDistribution"},
        }
        cfg["critic"] = {
            "class_name": "SAPGModel",
            "model_type": "rnn",
            "hidden_dims": [32],
            "rnn_type": "lstm",
            "rnn_hidden_dim": 16,
            "rnn_num_layers": 1,
        }
    else:
        cfg["actor"] = {
            "class_name": "SAPGModel",
            "model_type": "mlp",
            "hidden_dims": [32, 32],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution"},
        }
        cfg["critic"] = {
            "class_name": "SAPGModel",
            "model_type": "mlp",
            "hidden_dims": [32, 32],
            "activation": "elu",
        }
    return cfg


def _build_runner(model_type: str, aggregation: str) -> OnPolicyRunner:
    return OnPolicyRunner(DummyEnv(), _make_cfg(model_type, aggregation), log_dir=None, device="cpu")


@pytest.mark.parametrize("model_type", ["mlp", "rnn"])
@pytest.mark.parametrize("aggregation", ["none", "leader_follower", "symmetric"])
def test_sapg_learn_runs_and_updates(model_type: str, aggregation: str) -> None:
    """A short SAPG learn call should complete and update actor parameters."""
    runner = _build_runner(model_type, aggregation)
    before = {n: p.clone() for n, p in runner.alg.actor.named_parameters()}
    runner.learn(num_learning_iterations=2)
    changed = any(not torch.equal(before[n], p) for n, p in runner.alg.actor.named_parameters())
    assert changed, f"SAPG actor params should change ({model_type}, {aggregation})"


def test_sapg_requires_divisible_num_blocks() -> None:
    """An indivisible (num_envs, num_blocks) pair should raise on the first update."""
    cfg = _make_cfg("mlp", "leader_follower")
    cfg["algorithm"]["num_blocks"] = 3  # 8 % 3 != 0
    runner = OnPolicyRunner(DummyEnv(), cfg, log_dir=None, device="cpu")
    with pytest.raises(ValueError):
        runner.learn(num_learning_iterations=1)


def test_sapg_recurrent_inference_shape() -> None:
    """Inference policy from a recurrent SAPG runner returns correct action shape."""
    runner = _build_runner("rnn", "leader_follower")
    runner.learn(num_learning_iterations=1)
    policy = runner.get_inference_policy()
    policy.reset()  # clear the inference-mode hidden state left over from the rollout
    obs = runner.env.get_observations()
    obs["sapg_latent"] = torch.zeros(NUM_ENVS, 8)
    with torch.inference_mode():
        actions = policy(obs)
    assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
