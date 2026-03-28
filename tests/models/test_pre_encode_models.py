# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PreEncodeMLPModel and PreEncodeRecurrentModel."""

from __future__ import annotations

import tempfile
import torch
from tensordict import TensorDict

import onnx
import pytest

from rsl_rl.models import PreEncodeMLPModel, PreEncodeRecurrentModel

NUM_ENVS = 2
RAW_TACTILE_DIM = 100
ENCODED_TACTILE_DIM = 8
PROPRIO_DIM = 16
NUM_ACTIONS = 4
RNN_HIDDEN = 16

ENCODER_CFG = {
    "tactile": {"hidden_dims": [64, 32], "output_dim": ENCODED_TACTILE_DIM},
}


def _make_pre_encode_mlp(**kwargs: object) -> tuple[PreEncodeMLPModel, TensorDict]:
    obs = TensorDict(
        {
            "tactile": torch.randn(NUM_ENVS, RAW_TACTILE_DIM),
            "proprioception": torch.randn(NUM_ENVS, PROPRIO_DIM),
        },
        batch_size=[NUM_ENVS],
    )
    obs_groups = {"actor": ["tactile", "proprioception"]}
    defaults: dict[str, object] = dict(
        hidden_dims=[32, 32],
        encoder_cfg={"tactile": dict(ENCODER_CFG["tactile"])},
    )
    defaults.update(kwargs)
    model = PreEncodeMLPModel(obs, obs_groups, "actor", NUM_ACTIONS, **defaults)
    return model, obs


def _make_pre_encode_recurrent(**kwargs: object) -> tuple[PreEncodeRecurrentModel, TensorDict]:
    obs = TensorDict(
        {
            "tactile": torch.randn(NUM_ENVS, RAW_TACTILE_DIM),
            "proprioception": torch.randn(NUM_ENVS, PROPRIO_DIM),
        },
        batch_size=[NUM_ENVS],
    )
    obs_groups = {"actor": ["tactile", "proprioception"]}
    defaults: dict[str, object] = dict(
        hidden_dims=[32, 32],
        encoder_cfg={"tactile": dict(ENCODER_CFG["tactile"])},
        rnn_type="gru",
        rnn_hidden_dim=RNN_HIDDEN,
        rnn_num_layers=1,
    )
    defaults.update(kwargs)
    model = PreEncodeRecurrentModel(obs, obs_groups, "actor", NUM_ACTIONS, **defaults)
    return model, obs


class TestPreEncodeMLPLatent:
    def test_latent_dim_is_pass_plus_encoded(self) -> None:
        model, obs = _make_pre_encode_mlp()
        latent = model.get_latent(obs)
        assert latent.shape == (NUM_ENVS, PROPRIO_DIM + ENCODED_TACTILE_DIM)

    def test_pass_then_encode_order(self) -> None:
        model, obs = _make_pre_encode_mlp()
        latent = model.get_latent(obs)
        expected_pass = model.obs_normalizer(obs["proprioception"])
        assert torch.allclose(latent[:, :PROPRIO_DIM], expected_pass, atol=1e-6)

    def test_encoded_portion_changes_with_tactile_only(self) -> None:
        model, obs = _make_pre_encode_mlp()
        latent_before = model.get_latent(obs).detach().clone()
        obs["tactile"] = torch.randn_like(obs["tactile"])
        latent_after = model.get_latent(obs).detach()
        assert torch.allclose(latent_before[:, :PROPRIO_DIM], latent_after[:, :PROPRIO_DIM], atol=1e-6)
        assert not torch.allclose(latent_before[:, PROPRIO_DIM:], latent_after[:, PROPRIO_DIM:], atol=1e-6)

    def test_all_encoded_no_pass_through(self) -> None:
        obs = TensorDict({"tactile": torch.randn(NUM_ENVS, RAW_TACTILE_DIM)}, batch_size=[NUM_ENVS])
        obs_groups = {"actor": ["tactile"]}
        model = PreEncodeMLPModel(
            obs,
            obs_groups,
            "actor",
            NUM_ACTIONS,
            hidden_dims=[16],
            encoder_cfg={"tactile": dict(ENCODER_CFG["tactile"])},
        )
        latent = model.get_latent(obs)
        assert latent.shape == (NUM_ENVS, ENCODED_TACTILE_DIM)


class TestPreEncodeMLPSharing:
    def test_shared_encoders_same_object(self) -> None:
        model_a, obs = _make_pre_encode_mlp()
        model_b = PreEncodeMLPModel(
            obs,
            {"actor": ["tactile", "proprioception"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[64],
            encoders=model_a.encoders,
        )
        assert model_a.encoders["tactile"] is model_b.encoders["tactile"]

    def test_gradient_flows_to_shared_encoder(self) -> None:
        model_a, obs = _make_pre_encode_mlp()
        model_b = PreEncodeMLPModel(
            obs,
            {"actor": ["tactile", "proprioception"]},
            "actor",
            1,
            hidden_dims=[16],
            encoders=model_a.encoders,
        )
        model_b(obs).sum().backward()
        for p in model_a.encoders.parameters():
            assert p.grad is not None


class TestPreEncodeMLPErrors:
    def test_encoder_key_not_in_obs_set(self) -> None:
        obs = TensorDict({"proprioception": torch.randn(NUM_ENVS, PROPRIO_DIM)}, batch_size=[NUM_ENVS])
        with pytest.raises(ValueError, match="encoder_cfg contains keys not in this observation set"):
            PreEncodeMLPModel(
                obs,
                {"actor": ["proprioception"]},
                "actor",
                NUM_ACTIONS,
                encoder_cfg={"tactile": dict(ENCODER_CFG["tactile"])},
            )

    def test_extra_encoder_cfg_key(self) -> None:
        obs = TensorDict(
            {
                "tactile": torch.randn(NUM_ENVS, RAW_TACTILE_DIM),
                "proprioception": torch.randn(NUM_ENVS, PROPRIO_DIM),
            },
            batch_size=[NUM_ENVS],
        )
        bad_cfg = {
            "tactile": dict(ENCODER_CFG["tactile"]),
            "ghost": {"hidden_dims": [4], "output_dim": 2},
        }
        with pytest.raises(ValueError, match="encoder_cfg contains keys"):
            PreEncodeMLPModel(
                obs,
                {"actor": ["tactile", "proprioception"]},
                "actor",
                NUM_ACTIONS,
                encoder_cfg=bad_cfg,
            )


@pytest.mark.filterwarnings("ignore:.*legacy TorchScript.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
class TestPreEncodeMLPJIT:
    def test_jit_matches_eager(self) -> None:
        model, obs = _make_pre_encode_mlp(
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )
        model.eval()
        eager = model(obs).detach()
        jit_m = torch.jit.script(model.as_jit())
        obs_pass = obs["proprioception"]
        out = jit_m(obs_pass, [obs["tactile"]])
        assert torch.allclose(eager, out, atol=1e-5)


class TestPreEncodeMLPONNX:
    def test_onnx_export_valid(self) -> None:
        model, _obs = _make_pre_encode_mlp(
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )
        model.eval()
        onnx_model = model.as_onnx(verbose=False)
        onnx_model.eval()
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            torch.onnx.export(
                onnx_model,
                onnx_model.get_dummy_inputs(),
                f.name,
                export_params=True,
                opset_version=18,
                input_names=onnx_model.input_names,
                output_names=onnx_model.output_names,
            )
            loaded = onnx.load(f.name)
            onnx.checker.check_model(loaded)
        assert onnx_model.input_names == ["obs_pass", "tactile"]


class TestPreEncodeRecurrentLatent:
    def test_get_latent_shape_matches_rnn_hidden(self) -> None:
        model, obs = _make_pre_encode_recurrent()
        latent = model.get_latent(obs)
        assert latent.shape == (NUM_ENVS, RNN_HIDDEN)

    def test_rnn_input_dim_is_pass_plus_encoded(self) -> None:
        model, _obs = _make_pre_encode_recurrent()
        assert model.rnn.rnn.input_size == PROPRIO_DIM + ENCODED_TACTILE_DIM

    def test_hidden_state_accumulates(self) -> None:
        model, obs = _make_pre_encode_recurrent()
        out_fresh = model(obs).detach().clone()
        model.reset()
        for _ in range(3):
            model(obs)
        out_ctx = model(obs).detach()
        assert not torch.allclose(out_fresh, out_ctx, atol=1e-5)


class TestPreEncodeRecurrentExport:
    def test_jit_not_implemented(self) -> None:
        model, _obs = _make_pre_encode_recurrent()
        with pytest.raises(NotImplementedError, match="TorchScript"):
            model.as_jit()

    def test_onnx_not_implemented(self) -> None:
        model, _obs = _make_pre_encode_recurrent()
        with pytest.raises(NotImplementedError, match="ONNX"):
            model.as_onnx()
