# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SAPG: Split and Aggregate Policy Gradients (Singla et al., ICML 2024).

SAPG improves PPO in the large-batch regime (tens of thousands of parallel envs)
by *splitting* the environments into ``M`` blocks and treating each block as a
distinct policy, then *aggregating* their experience into a single update.

This implementation realizes the "``M`` policies" with a **single shared backbone**
conditioned on a per-block latent code (:mod:`rsl_rl.models.sapg_model`):

* **Split / diversity (Sec. 4.4-4.5).** Each block is conditioned on a distinct
  sinusoidal latent and trained with its own entropy coefficient, so the blocks
  explore differently while sharing weights.
* **Aggregate (Sec. 4.1-4.3).** After the rollout, data from "follower" blocks is
  relabeled with another block's latent, its values/returns are recomputed under
  that latent (1-step TD target, Eq. 6), and it is appended to the update batch as
  off-policy data. The standard PPO clip ratio — current policy under the new
  latent vs. the behavior log-prob — supplies the importance weighting (Eq. 3), so
  no explicit importance ratio term is needed. The off-policy contribution is
  weighted by ``off_policy_loss_coef`` (``lambda`` in Eq. 4).

Two aggregation schemes (Sec. 4.2-4.3) are supported:

* ``"leader_follower"`` — block 0 is the leader; it is updated with its own data
  plus relabeled data from a random subset of follower blocks. Followers train on
  their own on-policy data only. This is the variant in the paper's Fig. 3.
* ``"symmetric"`` — every block is additionally updated with relabeled data from a
  random subset of the other blocks. (More expensive: off-policy volume scales
  with ``M``.)
* ``"none"`` — pure PPO with latent conditioning and per-block entropy (ablation).

The whole algorithm is self-contained in this module plus
:mod:`rsl_rl.models.sapg_model` and :mod:`rsl_rl.storage.sapg_rollout_storage`;
the shared PPO / ``RolloutStorage`` code is untouched.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models.sapg_model import SAPG_LATENT_KEY
from rsl_rl.storage.sapg_rollout_storage import (
    SapgBatchBundle,
    sapg_mini_batch_generator,
    sapg_recurrent_mini_batch_generator,
)


def create_sinusoidal_encoding(values: torch.Tensor, dim: int, n: float = 10.0) -> torch.Tensor:
    """Return a ``[len(values), dim]`` sinusoidal encoding of a 1-D tensor.

    Mirrors the encoding used by the reference SAPG implementation to turn a scalar
    per-block identifier into a smooth, distinguishable latent code.
    """
    if dim % 2 != 0:
        raise ValueError(f"SAPG latent_dim must be even, got {dim}.")
    denom = n ** (2 * torch.arange(dim // 2, dtype=torch.float32, device=values.device) / dim)
    angles = values.unsqueeze(-1) / denom
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class SAPG(PPO):
    """Split and Aggregate Policy Gradients on top of :class:`~rsl_rl.algorithms.PPO`."""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        storage,
        num_blocks: int = 4,
        aggregation: str = "leader_follower",
        off_policy_ratio: int = 1,
        off_policy_loss_coef: float = 1.0,
        latent_dim: int = 8,
        latent_scale: float = 50.0,
        latent_n: float = 10.0,
        entropy_coef_range: float = 0.0,
        **ppo_kwargs,
    ) -> None:
        """Initialize SAPG.

        Args:
            num_blocks: Number of policy blocks ``M``. ``num_envs`` must be divisible by it.
            aggregation: One of ``"leader_follower"``, ``"symmetric"`` or ``"none"``.
            off_policy_ratio: Number of follower blocks aggregated per leader, per update.
            off_policy_loss_coef: Weight ``lambda`` on the off-policy surrogate term (Eq. 4).
            latent_dim: Dimension of the per-block latent code (even).
            latent_scale: Spread of the per-block scalar fed to the sinusoidal encoding.
            latent_n: Base of the sinusoidal encoding.
            entropy_coef_range: Per-block entropy spread. Block 0 (leader) uses the base
                ``entropy_coef``; the last block uses ``entropy_coef + entropy_coef_range``.
            ppo_kwargs: Forwarded to :class:`~rsl_rl.algorithms.PPO`.
        """
        super().__init__(actor, critic, storage, **ppo_kwargs)

        if aggregation not in ("leader_follower", "symmetric", "none"):
            raise ValueError(f"Unknown SAPG aggregation '{aggregation}'.")
        if self.rnd is not None or self.symmetry is not None:
            raise ValueError("SAPG does not support the RND or symmetry extensions.")

        self.num_blocks = num_blocks
        self.aggregation = aggregation
        self.off_policy_ratio = off_policy_ratio
        self.off_policy_loss_coef = off_policy_loss_coef
        self.latent_dim = latent_dim
        self.latent_scale = latent_scale
        self.latent_n = latent_n
        self.entropy_coef_range = entropy_coef_range

        self._blocks_ready = False
        self._final_obs: TensorDict | None = None

    # ------------------------------------------------------------------ #
    # Block / latent setup                                               #
    # ------------------------------------------------------------------ #
    def _setup_blocks(self) -> None:
        """Build the per-block latents and per-environment latent / entropy tables."""
        num_envs = self.storage.num_envs
        if num_envs % self.num_blocks != 0:
            raise ValueError(
                f"SAPG requires num_envs ({num_envs}) divisible by num_blocks ({self.num_blocks})."
            )
        self.block_size = num_envs // self.num_blocks
        device = self.device

        block_id = torch.arange(self.num_blocks, device=device).repeat_interleave(self.block_size)
        self.env_block_id = block_id  # [num_envs]

        # Per-block scalar -> sinusoidal latent code. Block 0 (leader) is the
        # "exploiter"; later blocks get a higher entropy coefficient (explore more).
        block_scalar = torch.linspace(self.latent_scale, 0.0, self.num_blocks, device=device)
        self.block_latent = create_sinusoidal_encoding(block_scalar, self.latent_dim, self.latent_n)  # [M, D]
        self.env_latent = self.block_latent[block_id]  # [num_envs, D]

        block_entropy = self.entropy_coef + torch.linspace(
            0.0, self.entropy_coef_range, self.num_blocks, device=device
        )
        self.block_entropy_coef = block_entropy  # [M]
        self.env_entropy_coef = block_entropy[block_id]  # [num_envs]

        self._blocks_ready = True

    def _inject_latent(self, obs: TensorDict) -> TensorDict:
        """Write the per-environment latent code into ``obs`` (in place) and return it."""
        if not self._blocks_ready:
            self._setup_blocks()
        obs[SAPG_LATENT_KEY] = self.env_latent
        return obs

    # ------------------------------------------------------------------ #
    # Rollout hooks                                                      #
    # ------------------------------------------------------------------ #
    def act(self, obs: TensorDict) -> torch.Tensor:
        """Condition the observation on the per-environment latent before acting."""
        return super().act(self._inject_latent(obs))

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute GAE returns/advantages (latent-conditioned), deferring normalization.

        Advantage normalization is deferred to after off-policy augmentation so the
        on-policy and off-policy advantages are normalized together.
        """
        self._inject_latent(obs)
        self._final_obs = obs  # cached for the off-policy 1-step-TD bootstrap value

        st = self.storage
        last_values = self.critic(obs).detach()
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values  # raw; normalized post-augmentation

    # ------------------------------------------------------------------ #
    # Off-policy aggregation                                             #
    # ------------------------------------------------------------------ #
    def _select_aggregation_pairs(self) -> list[tuple[int, int]]:
        """Return a list of ``(target_block, source_block)`` relabeling pairs.

        The target block's latent is imposed on the source block's data. The choice
        is randomized each update and broadcast across ranks under multi-GPU.
        """
        m = self.num_blocks
        pairs: list[tuple[int, int]] = []
        if self.aggregation == "leader_follower":
            others = torch.randperm(m - 1) + 1  # follower blocks 1..M-1
            k = min(m - 1, self.off_policy_ratio)
            pairs = [(0, int(j)) for j in others[:k]]
        elif self.aggregation == "symmetric":
            for i in range(m):
                others = torch.tensor([b for b in range(m) if b != i])
                perm = others[torch.randperm(m - 1)]
                k = min(m - 1, self.off_policy_ratio)
                pairs += [(i, int(j)) for j in perm[:k]]

        if self.is_multi_gpu:
            obj = [pairs]
            torch.distributed.broadcast_object_list(obj, src=0)
            pairs = obj[0]
        return pairs

    def _block_env_ids(self, block: int) -> torch.Tensor:
        """Return the environment indices belonging to ``block``."""
        start = block * self.block_size
        return torch.arange(start, start + self.block_size, device=self.device)

    def _slice_critic_hidden(self, step: int, env_ids: torch.Tensor):
        """Return the saved critic hidden state at ``step`` for ``env_ids`` (RNN only)."""
        saved = self.storage.saved_hidden_state_c
        if saved is None:
            return None
        states = [buf[step][:, env_ids].contiguous() for buf in saved]  # each [num_layers, bs, hidden]
        return states[0] if len(states) == 1 else tuple(states)

    @torch.no_grad()
    def _recompute_block_values(
        self, env_ids: torch.Tensor, relabeled_obs: TensorDict, leader_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute per-step values and the bootstrap value under the imposed latent.

        Returns ``(values[T, bs, 1], last_value[bs, 1])``.
        """
        st = self.storage
        t = st.num_transitions_per_env
        if self.critic.is_recurrent:
            saved_hidden = self.critic.get_hidden_state()  # preserve rollout continuity
            self.critic.reset(hidden_state=self._slice_critic_hidden(0, env_ids))
            values = []
            for step in range(t):
                values.append(self.critic(relabeled_obs[step]).detach())
                self.critic.reset(st.dones[step, env_ids].squeeze(-1))
            values = torch.stack(values, dim=0)
            last_obs = self._final_obs[env_ids].clone()
            last_obs[SAPG_LATENT_KEY] = leader_latent.expand_as(last_obs[SAPG_LATENT_KEY])
            last_value = self.critic(last_obs).detach()
            self.critic.reset(hidden_state=saved_hidden)
        else:
            flat = relabeled_obs.flatten(0, 1)
            values = self.critic(flat).detach().view(t, env_ids.shape[0], 1)
            last_obs = self._final_obs[env_ids].clone()
            last_obs[SAPG_LATENT_KEY] = leader_latent.expand_as(last_obs[SAPG_LATENT_KEY])
            last_value = self.critic(last_obs).detach()
        return values, last_value

    def _build_bundle(self) -> SapgBatchBundle:
        """Assemble the (possibly augmented) batch bundle for the update."""
        st = self.storage
        t, n = st.num_transitions_per_env, st.num_envs

        # On-policy base.
        obs_list = [st.observations]
        actions_list = [st.actions]
        dones_list = [st.dones]
        values_list = [st.values]
        logp_list = [st.actions_log_prob]
        returns_list = [st.returns]
        adv_list = [st.advantages]
        dist_lists = [[p] for p in st.distribution_params]
        ec_list = [self.env_entropy_coef.view(1, n, 1).expand(t, n, 1)]
        ow_list = [torch.zeros(t, n, 1, device=self.device)]
        ha_lists = [[buf] for buf in st.saved_hidden_state_a] if st.saved_hidden_state_a is not None else None
        hc_lists = [[buf] for buf in st.saved_hidden_state_c] if st.saved_hidden_state_c is not None else None

        # Off-policy relabeled blocks.
        for target_block, source_block in self._select_aggregation_pairs():
            env_ids = self._block_env_ids(source_block)
            leader_latent = self.block_latent[target_block]

            obs = st.observations[:, env_ids].clone()
            obs[SAPG_LATENT_KEY] = leader_latent.expand_as(obs[SAPG_LATENT_KEY])
            values, last_value = self._recompute_block_values(env_ids, obs, leader_latent)

            dones = st.dones[:, env_ids]
            rewards = st.rewards[:, env_ids]
            values_plus = torch.cat([values, last_value.unsqueeze(0)], dim=0)
            # 1-step TD target under the imposed latent (Eq. 6).
            returns = rewards + self.gamma * values_plus[1:] * (1.0 - dones.float())
            advantages = returns - values

            obs_list.append(obs)
            actions_list.append(st.actions[:, env_ids])
            dones_list.append(dones)
            values_list.append(values)
            logp_list.append(st.actions_log_prob[:, env_ids])
            returns_list.append(returns)
            adv_list.append(advantages)
            for k, p in enumerate(st.distribution_params):
                dist_lists[k].append(p[:, env_ids])
            bs = env_ids.shape[0]
            ec_list.append(torch.full((t, bs, 1), float(self.block_entropy_coef[target_block]), device=self.device))
            ow_list.append(torch.ones(t, bs, 1, device=self.device))
            if ha_lists is not None:
                for k, buf in enumerate(st.saved_hidden_state_a):
                    ha_lists[k].append(buf[:, :, env_ids])
            if hc_lists is not None:
                for k, buf in enumerate(st.saved_hidden_state_c):
                    hc_lists[k].append(buf[:, :, env_ids])

        observations = torch.cat(obs_list, dim=1)
        advantages = torch.cat(adv_list, dim=1)
        if not self.normalize_advantage_per_mini_batch:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return SapgBatchBundle(
            observations=observations,
            actions=torch.cat(actions_list, dim=1),
            dones=torch.cat(dones_list, dim=1),
            values=torch.cat(values_list, dim=1),
            actions_log_prob=torch.cat(logp_list, dim=1),
            distribution_params=tuple(torch.cat(parts, dim=1) for parts in dist_lists),
            returns=torch.cat(returns_list, dim=1),
            advantages=advantages,
            entropy_coef=torch.cat(ec_list, dim=1),
            offpolicy_mask=torch.cat(ow_list, dim=1),
            saved_hidden_state_a=[torch.cat(parts, dim=2) for parts in ha_lists] if ha_lists is not None else None,
            saved_hidden_state_c=[torch.cat(parts, dim=2) for parts in hc_lists] if hc_lists is not None else None,
        )

    # ------------------------------------------------------------------ #
    # Update                                                            #
    # ------------------------------------------------------------------ #
    def update(self) -> dict[str, float]:
        """Run SAPG optimization epochs over the aggregated batch."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_offpolicy_frac = 0.0

        bundle = self._build_bundle()
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = sapg_recurrent_mini_batch_generator(bundle, self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = sapg_mini_batch_generator(bundle, self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            # Recompute log-probs, values and entropy under the current parameters.
            self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy

            # Adaptive learning-rate schedule (KL over the full aggregated batch).
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            advantages = torch.squeeze(batch.advantages, -1)
            old_log_prob = torch.squeeze(batch.old_actions_log_prob, -1)
            offpolicy = torch.squeeze(batch.offpolicy_mask, -1) > 0.5
            entropy_coef = torch.squeeze(batch.entropy_coef, -1)

            # Surrogate (clipped) objective. The PPO clip ratio between the current
            # policy (under the imposed latent) and the behavior log-prob is exactly
            # the off-policy importance weight (Eq. 3), so off-policy samples enter
            # the same surrogate, weighted by ``off_policy_loss_coef`` (Eq. 4).
            ratio = torch.exp(actions_log_prob - old_log_prob)
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            per_sample_surrogate = torch.max(surrogate, surrogate_clipped)
            on_term = per_sample_surrogate[~offpolicy].mean() if (~offpolicy).any() else 0.0
            off_term = per_sample_surrogate[offpolicy].mean() if offpolicy.any() else 0.0
            surrogate_loss = on_term + self.off_policy_loss_coef * off_term

            # Value loss (clipped), over all samples.
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            # Per-block entropy regularization.
            entropy_loss = (entropy_coef * entropy).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.detach().item() if torch.is_tensor(surrogate_loss) else float(surrogate_loss)
            mean_entropy += entropy.mean().item()
            mean_offpolicy_frac += offpolicy.float().mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_offpolicy_frac /= num_updates

        self.storage.clear()
        self._final_obs = None

        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "offpolicy_frac": mean_offpolicy_frac,
        }

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> "SAPG":
        """Construct SAPG, wiring the latent code through obs, storage and models."""
        latent_dim = cfg["algorithm"].get("latent_dim", 8)
        # Allocate the latent observation key so the storage buffers and the model
        # input dimensions account for it from construction time onward.
        obs[SAPG_LATENT_KEY] = torch.zeros(obs.batch_size[0], latent_dim, device=obs.device)
        cfg["actor"]["latent_dim"] = latent_dim
        cfg["critic"]["latent_dim"] = latent_dim
        return PPO.construct_algorithm(obs, env, cfg, device)
