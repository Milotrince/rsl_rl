# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Distillation algorithm."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.distillation import AuxiliaryLoss, Distillation
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 12
OBS_DIM = 8
NUM_ACTIONS = 4
AUX_DIM = 2


class DummyAuxiliaryLoss(AuxiliaryLoss):
    """Small auxiliary head used to exercise the generic distillation hook."""

    def __init__(self, target_obs: str, weight: float = 1.0, name: str = "dummy") -> None:
        """Store config used during setup and compute."""
        super().__init__()
        self.target_obs = target_obs
        self.weight = weight
        self.name = name
        self.head: nn.Linear | None = None

    def setup(
        self,
        student: MLPModel,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        device: str,
    ) -> list[nn.Parameter]:
        """Build a prediction head from the student's latent to the target observation."""
        assert self.target_obs not in obs_groups["student"]
        assert self.target_obs not in obs_groups["teacher"]
        latent = student.get_latent(obs)
        self.head = nn.Linear(latent.shape[-1], obs[self.target_obs].shape[-1]).to(device)
        return list(self.head.parameters())

    def compute(self, batch: RolloutStorage.Batch, student: MLPModel) -> dict[str, torch.Tensor]:
        """Predict the configured target observation from the student latent."""
        assert self.head is not None
        latent = student.get_latent(batch.observations)
        prediction = self.head(latent)
        target = batch.observations[self.target_obs]
        return {self.name: self.weight * nn.functional.mse_loss(prediction, target)}


class _DummyDistillationEnv:
    """Minimal environment shell for Distillation.construct_algorithm."""

    num_envs = NUM_ENVS
    num_actions = NUM_ACTIONS


def _make_distillation_setup(gradient_length: int = 3, num_learning_epochs: int = 1) -> tuple:
    """Build a Distillation instance with small networks."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"student": ["policy"], "teacher": ["policy"]}

    student = MLPModel(obs, obs_groups, "student", NUM_ACTIONS, hidden_dims=[32, 32])
    teacher = MLPModel(obs, obs_groups, "teacher", NUM_ACTIONS, hidden_dims=[32, 32])

    storage = RolloutStorage("distillation", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

    alg = Distillation(
        student,
        teacher,
        storage,
        num_learning_epochs=num_learning_epochs,
        gradient_length=gradient_length,
        learning_rate=1e-3,
    )
    return alg, obs, storage


def _make_aux_obs() -> TensorDict:
    """Create observations with an auxiliary target hidden from student/teacher groups."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs["aux_target"] = obs["policy"][:, :AUX_DIM].clone()
    return obs


def _make_distillation_cfg_with_auxiliary_loss() -> dict:
    """Return a minimal distillation config with a dummy auxiliary loss."""
    return {
        "num_steps_per_env": NUM_STEPS,
        "obs_groups": {"student": ["policy"], "teacher": ["policy"]},
        "student": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
        },
        "teacher": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
        },
        "algorithm": {
            "class_name": "Distillation",
            "num_learning_epochs": 1,
            "gradient_length": 3,
            "learning_rate": 1e-3,
            "auxiliary_losses": [
                {
                    "class_name": "tests.algorithms.test_distillation:DummyAuxiliaryLoss",
                    "target_obs": "aux_target",
                    "weight": 0.5,
                    "name": "dummy",
                }
            ],
        },
        "multi_gpu": None,
    }


def _fill_distillation_storage(alg: Distillation, obs: TensorDict) -> None:
    """Fill the distillation storage with transitions."""
    for _ in range(NUM_STEPS):
        t = RolloutStorage.Transition()
        t.observations = obs
        t.hidden_states = (None, None)
        t.actions = alg.student(obs).detach()
        t.privileged_actions = alg.teacher(obs).detach()
        t.rewards = torch.randn(NUM_ENVS)
        t.dones = torch.zeros(NUM_ENVS)
        alg.storage.add_transition(t)


class TestDistillationLoss:
    """Tests for distillation loss computation."""

    def test_loss_decreases_over_updates(self) -> None:
        """Behavior loss should decrease over repeated update() calls (learning signal works)."""
        alg, obs, _storage = _make_distillation_setup(gradient_length=3, num_learning_epochs=2)
        alg.train_mode()

        losses = []
        for _ in range(5):
            _fill_distillation_storage(alg, obs)
            loss_dict = alg.update()
            losses.append(loss_dict["behavior"])

        # Loss should generally decrease; allow some noise — check first vs last
        assert losses[-1] < losses[0], f"Loss should decrease over updates, got {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_gradient_accumulation_step_count(self) -> None:
        """Optimizer should step floor(num_transitions / gradient_length) times per epoch."""
        gradient_length = 4
        alg, obs, _storage = _make_distillation_setup(gradient_length=gradient_length, num_learning_epochs=1)
        alg.train_mode()

        _fill_distillation_storage(alg, obs)

        step_count = 0
        original_step = alg.optimizer.step

        def counting_step(*args: object, **kwargs: object) -> None:
            nonlocal step_count
            step_count += 1
            return original_step(*args, **kwargs)

        alg.optimizer.step = counting_step
        alg.update()

        expected_steps = NUM_STEPS // gradient_length
        assert step_count == expected_steps, f"Expected {expected_steps} optimizer steps, got {step_count}"

    def test_update_changes_student_but_not_teacher(self) -> None:
        """Student parameters should change after update, while teacher parameters remain frozen."""
        alg, obs, _storage = _make_distillation_setup(gradient_length=3)
        alg.train_mode()

        student_before = {name: p.clone() for name, p in alg.student.named_parameters()}
        teacher_before = {name: p.clone() for name, p in alg.teacher.named_parameters()}

        _fill_distillation_storage(alg, obs)
        alg.update()

        any_student_changed = any(
            not torch.equal(p, student_before[name]) for name, p in alg.student.named_parameters()
        )
        assert any_student_changed, "Student parameters should change after an update"

        for name, p in alg.teacher.named_parameters():
            assert torch.equal(p, teacher_before[name]), f"Teacher parameter {name} changed during student update"


class TestDistillationAuxiliaryLoss:
    """Tests for the generic distillation auxiliary-loss hook."""

    def test_no_auxiliary_losses_keep_checkpoint_surface(self) -> None:
        """Checkpoints without auxiliary losses should not grow new auxiliary keys."""
        alg, _obs, _storage = _make_distillation_setup()

        assert "auxiliary_loss_state_dicts" not in alg.save()

    def test_configured_auxiliary_loss_updates_logs_and_checkpoints(self) -> None:
        """A configured auxiliary loss should optimize its parameters and report aux/<name>."""
        obs = _make_aux_obs()
        alg = Distillation.construct_algorithm(
            obs, _DummyDistillationEnv(), _make_distillation_cfg_with_auxiliary_loss(), "cpu"
        )
        alg.train_mode()

        assert len(alg.auxiliary_losses) == 1
        assert "aux_target" not in alg.student.obs_groups
        assert "aux_target" not in alg.teacher.obs_groups
        auxiliary_loss = alg.auxiliary_losses[0]
        auxiliary_before = copy.deepcopy(auxiliary_loss.state_dict())
        auxiliary_parameter = next(auxiliary_loss.parameters())
        optimizer_parameters = [param for group in alg.optimizer.param_groups for param in group["params"]]
        assert any(param is auxiliary_parameter for param in optimizer_parameters)

        _fill_distillation_storage(alg, obs)
        loss_dict = alg.update()

        assert "aux/dummy" in loss_dict
        assert any(
            not torch.equal(auxiliary_before[name], parameter)
            for name, parameter in auxiliary_loss.state_dict().items()
        )
        assert auxiliary_parameter in alg.optimizer.state

        checkpoint = alg.save()
        assert "auxiliary_loss_state_dicts" in checkpoint
        assert "head.weight" in checkpoint["auxiliary_loss_state_dicts"][0]

        restored_alg = Distillation.construct_algorithm(
            obs, _DummyDistillationEnv(), _make_distillation_cfg_with_auxiliary_loss(), "cpu"
        )
        restored_alg.load(checkpoint, load_cfg=None, strict=True)
        restored_auxiliary_state = restored_alg.auxiliary_losses[0].state_dict()
        for name, parameter in checkpoint["auxiliary_loss_state_dicts"][0].items():
            assert torch.equal(restored_auxiliary_state[name], parameter)
