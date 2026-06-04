# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mini-batch generators for SAPG's aggregated (on-policy + off-policy) batch.

After a rollout, SAPG augments the data along the *environment* axis: relabeled
off-policy trajectories from "follower" blocks are appended to the on-policy data
(see :class:`~rsl_rl.algorithms.sapg.SAPG`). The augmented batch carries two extra
per-sample channels that the standard :class:`~rsl_rl.storage.RolloutStorage`
generators do not produce:

* ``entropy_coef`` — the per-environment entropy coefficient (SAPG conditions each
  block on a different exploration level, Sec. 4.5).
* ``offpolicy_mask`` — ``1`` for relabeled off-policy samples, ``0`` for on-policy
  samples, used to apply the off-policy loss weight ``lambda`` (Eq. 4).

These generators mirror the feed-forward and recurrent generators of
``RolloutStorage`` exactly (same shuffling / trajectory-splitting / hidden-state
reshaping) but operate on a :class:`SapgBatchBundle` and attach the two channels
to each yielded :class:`~rsl_rl.storage.RolloutStorage.Batch`. They live here,
separate from the shared storage, so SAPG needs no edits to ``RolloutStorage``.
"""

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.modules import HiddenState
from rsl_rl.storage.rollout_storage import RolloutStorage
from rsl_rl.utils import split_and_pad_trajectories


class SapgBatchBundle:
    """A container for SAPG's augmented (on-policy + off-policy) rollout data.

    All tensors follow the ``[num_transitions_per_env, num_envs, ...]`` layout of
    :class:`~rsl_rl.storage.RolloutStorage`, where ``num_envs`` is the *augmented*
    environment count (original envs plus appended off-policy trajectories).
    """

    def __init__(
        self,
        observations: TensorDict,
        actions: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        actions_log_prob: torch.Tensor,
        distribution_params: tuple[torch.Tensor, ...],
        returns: torch.Tensor,
        advantages: torch.Tensor,
        entropy_coef: torch.Tensor,
        offpolicy_mask: torch.Tensor,
        saved_hidden_state_a: list[torch.Tensor] | None,
        saved_hidden_state_c: list[torch.Tensor] | None,
    ) -> None:
        """Store the augmented rollout tensors."""
        self.observations = observations
        self.actions = actions
        self.dones = dones
        self.values = values
        self.actions_log_prob = actions_log_prob
        self.distribution_params = distribution_params
        self.returns = returns
        self.advantages = advantages
        self.entropy_coef = entropy_coef
        self.offpolicy_mask = offpolicy_mask
        self.saved_hidden_state_a = saved_hidden_state_a
        self.saved_hidden_state_c = saved_hidden_state_c

        self.num_transitions_per_env = observations.shape[0]
        self.num_envs = observations.shape[1]


def sapg_mini_batch_generator(
    bundle: SapgBatchBundle, num_mini_batches: int, num_epochs: int = 8
) -> Generator[RolloutStorage.Batch, None, None]:
    """Yield shuffled flat mini-batches for feed-forward SAPG updates.

    Mirrors :meth:`RolloutStorage.mini_batch_generator` and additionally attaches
    the ``entropy_coef`` and ``offpolicy_mask`` channels to each batch.
    """
    batch_size = bundle.num_envs * bundle.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    indices = torch.randperm(
        num_mini_batches * mini_batch_size, requires_grad=False, device=bundle.observations.device
    )

    observations = bundle.observations.flatten(0, 1)
    actions = bundle.actions.flatten(0, 1)
    values = bundle.values.flatten(0, 1)
    returns = bundle.returns.flatten(0, 1)
    old_actions_log_prob = bundle.actions_log_prob.flatten(0, 1)
    advantages = bundle.advantages.flatten(0, 1)
    old_distribution_params = tuple(p.flatten(0, 1) for p in bundle.distribution_params)
    entropy_coef = bundle.entropy_coef.flatten(0, 1)
    offpolicy_mask = bundle.offpolicy_mask.flatten(0, 1)

    for _ in range(num_epochs):
        for i in range(num_mini_batches):
            start = i * mini_batch_size
            stop = (i + 1) * mini_batch_size
            batch_idx = indices[start:stop]

            batch = RolloutStorage.Batch(
                observations=observations[batch_idx],
                actions=actions[batch_idx],
                values=values[batch_idx],
                advantages=advantages[batch_idx],
                returns=returns[batch_idx],
                old_actions_log_prob=old_actions_log_prob[batch_idx],
                old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
            )
            batch.entropy_coef = entropy_coef[batch_idx]
            batch.offpolicy_mask = offpolicy_mask[batch_idx]
            yield batch


def sapg_recurrent_mini_batch_generator(
    bundle: SapgBatchBundle, num_mini_batches: int, num_epochs: int = 8
) -> Generator[RolloutStorage.Batch, None, None]:
    """Yield trajectory mini-batches with masks and hidden states for recurrent SAPG.

    Mirrors :meth:`RolloutStorage.recurrent_mini_batch_generator` and additionally
    attaches the ``entropy_coef`` and ``offpolicy_mask`` channels (sliced along the
    environment axis exactly like ``advantages`` so they stay aligned with the
    *unpadded* per-step model outputs).
    """
    padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(bundle.observations, bundle.dones)
    mini_batch_size = bundle.num_envs // num_mini_batches

    for _ in range(num_epochs):
        first_traj = 0
        for i in range(num_mini_batches):
            start = i * mini_batch_size
            stop = (i + 1) * mini_batch_size

            dones = bundle.dones.squeeze(-1)
            last_was_done = torch.zeros_like(dones, dtype=torch.bool)
            last_was_done[1:] = dones[:-1]
            last_was_done[0] = True
            trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
            last_traj = first_traj + trajectories_batch_size

            last_was_done = last_was_done.permute(1, 0)
            if bundle.saved_hidden_state_a is not None:
                hidden_state_a_batch = [
                    saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_state in bundle.saved_hidden_state_a
                ]
                hidden_state_a_batch = (
                    hidden_state_a_batch[0] if len(hidden_state_a_batch) == 1 else hidden_state_a_batch
                )
            else:
                hidden_state_a_batch = None
            if bundle.saved_hidden_state_c is not None:
                hidden_state_c_batch = [
                    saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_state in bundle.saved_hidden_state_c
                ]
                hidden_state_c_batch = (
                    hidden_state_c_batch[0] if len(hidden_state_c_batch) == 1 else hidden_state_c_batch
                )
            else:
                hidden_state_c_batch = None

            batch = RolloutStorage.Batch(
                observations=padded_obs_trajectories[:, first_traj:last_traj],
                actions=bundle.actions[:, start:stop],
                values=bundle.values[:, start:stop],
                advantages=bundle.advantages[:, start:stop],
                returns=bundle.returns[:, start:stop],
                old_actions_log_prob=bundle.actions_log_prob[:, start:stop],
                old_distribution_params=tuple(p[:, start:stop] for p in bundle.distribution_params),
                hidden_states=(hidden_state_a_batch, hidden_state_c_batch),
                masks=trajectory_masks[:, first_traj:last_traj],
            )
            batch.entropy_coef = bundle.entropy_coef[:, start:stop]
            batch.offpolicy_mask = bundle.offpolicy_mask[:, start:stop]
            yield batch

            first_traj = last_traj
