# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Latent-conditioned model for SAPG (Split and Aggregate Policy Gradients).

SAPG trains a *single* shared backbone that represents ``M`` distinct policies by
conditioning on a per-environment latent code (Singla et al., ICML 2024, Sec. 4.4).
:class:`SAPGModel` is a single model that can be either feed-forward (``model_type
== "mlp"``) or recurrent (``model_type == "rnn"``); both read the latent from a
reserved ``"sapg_latent"`` observation key and concatenate it to the (normalized)
observation *after* normalization.

Keeping the latent out of the normalizer matters: it is a fixed, near-constant
value within a block, so feeding it through the running mean/std statistics would
either be a no-op (it carries no information once whitened) or destabilize the
normalizer with a zero-variance feature.

The latent is threaded through the observation ``TensorDict`` (rather than passed
as a separate argument) so that it rides along automatically through the rollout
storage and the mini-batch generators — including the recurrent split-and-pad
path — with no changes to the shared storage code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import RNN, HiddenState

SAPG_LATENT_KEY = "sapg_latent"
"""Reserved observation key carrying the per-environment SAPG latent code."""


class SAPGModel(MLPModel):
    """Latent-conditioned actor/critic model for SAPG, MLP or RNN.

    The latent (read from ``obs[SAPG_LATENT_KEY]``) is concatenated to the
    normalized observation. For ``model_type == "mlp"`` it is fed straight into the
    MLP trunk (input dimension ``obs_dim + latent_dim``); for ``model_type ==
    "rnn"`` it is concatenated before the RNN (RNN input ``obs_dim + latent_dim``)
    so the recurrent dynamics are conditioned on the block identity.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        model_type: str = "mlp",
        latent_dim: int = 8,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
    ) -> None:
        """Initialize the model.

        Args:
            model_type: ``"mlp"`` (feed-forward) or ``"rnn"`` (recurrent).
            latent_dim: Width of the per-block latent code (must be even).
            rnn_type / rnn_hidden_dim / rnn_num_layers: RNN settings (used only when
                ``model_type == "rnn"``).
            Remaining arguments match :class:`~rsl_rl.models.MLPModel`.
        """
        self.model_type = model_type.lower()
        if self.model_type not in ("mlp", "rnn"):
            raise ValueError(f"SAPGModel model_type must be 'mlp' or 'rnn', got '{model_type}'.")
        # Instance attribute shadows the ``MLPModel.is_recurrent`` class attribute so
        # the algorithm/runner pick up the right recurrent behavior per instance.
        self.is_recurrent = self.model_type == "rnn"
        self.sapg_latent_dim = latent_dim
        if self.is_recurrent:
            self.latent_dim = rnn_hidden_dim  # MLP head consumes the RNN output

        # Stored before ``super().__init__`` since it sizes the MLP via ``_get_latent_dim``.
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg)

        if self.is_recurrent:
            self.rnn = RNN(self.obs_dim + latent_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)

    def _get_latent_dim(self) -> int:
        """Return the MLP-trunk input dimension."""
        if self.is_recurrent:
            return self.latent_dim  # RNN hidden size
        return self.obs_dim + self.sapg_latent_dim

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Concatenate the SAPG latent to the normalized observation; run the RNN if recurrent."""
        latent = torch.cat([super().get_latent(obs), obs[SAPG_LATENT_KEY]], dim=-1)
        if self.is_recurrent:
            latent = self.rnn(latent, masks, hidden_state).squeeze(0)
        return latent

    # -- Recurrent hidden-state hooks (no-ops in MLP mode) -------------------- #
    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the RNN hidden state (no-op for MLP)."""
        if self.is_recurrent:
            self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        """Return the RNN hidden state (``None`` for MLP)."""
        return self.rnn.hidden_state if self.is_recurrent else None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the RNN hidden state for truncated backpropagation (no-op for MLP)."""
        if self.is_recurrent:
            self.rnn.detach_hidden_state(dones)

    # -- Export ------------------------------------------------------------- #
    def as_jit(self) -> nn.Module:
        """SAPG models read the latent from the observation dict and cannot use the standard exporters."""
        raise NotImplementedError(
            "SAPGModel conditions on a latent passed through the observation dict and cannot be exported with the"
            " standard observation-only JIT exporter."
        )

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """SAPG models read the latent from the observation dict and cannot use the standard exporters."""
        raise NotImplementedError(
            "SAPGModel conditions on a latent passed through the observation dict and cannot be exported with the"
            " standard observation-only ONNX exporter."
        )
