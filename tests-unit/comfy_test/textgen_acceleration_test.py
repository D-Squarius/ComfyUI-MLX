import json
import types

import pytest

from comfy.text_encoders import acceleration


def shape(width):
    return types.SimpleNamespace(shape=[width])


def test_acceleration_sidecar_defaults_to_no_behavior_without_file(tmp_path):
    model = tmp_path / "gemma_4_E4B_it.safetensors"
    model.write_bytes(b"")

    sidecar, config = acceleration.load_acceleration_config_for_path(str(model))

    assert sidecar is None
    assert config is None


def test_acceleration_sidecar_parses_gemma4_assistant_pair(tmp_path):
    model = tmp_path / "gemma_4_E4B_it.safetensors"
    model.write_bytes(b"")
    (tmp_path / "gemma_4_E4B_it.accel.json").write_text(
        json.dumps(
            {
                "acceleration": {
                    "mode": "auto",
                    "family": "gemma4_e4b",
                    "backend": "torch_mps",
                    "target_device": "gpu",
                    "target_initial_device": "gpu",
                    "target_dtype": "float16",
                    "assistant_clip_name": "gemma_4_E4B_it_assistant.safetensors",
                    "assistant_kind": "gemma4_assistant",
                    "assistant_repo_id": "google/gemma-4-E4B-it-assistant",
                    "assistant_revision": "main",
                    "assistant_dtype": "float16",
                    "max_speculative_tokens": 4,
                    "strict": True,
                }
            }
        ),
        encoding="utf-8",
    )

    sidecar, config = acceleration.load_acceleration_config_for_path(str(model))

    assert sidecar.endswith("gemma_4_E4B_it.accel.json")
    assert config.mode == "auto"
    assert config.strict is True
    assert config.family == "gemma4_e4b"
    assert config.target_device == "gpu"
    assert config.target_initial_device == "gpu"
    assert config.target_dtype == "float16"
    assert config.assistant_clip_name == "gemma_4_E4B_it_assistant.safetensors"
    assert config.assistant_kind == "gemma4_assistant"
    assert config.assistant_repo_id == "google/gemma-4-E4B-it-assistant"
    assert config.assistant_dtype == "float16"
    assert config.max_speculative_tokens == 4


def test_detects_gemma4_and_qwen35_families_from_state_dict():
    assert acceleration.detect_text_encoder_family({"model.layers.41.self_attn.q_norm.weight": shape(1), "model.layers.0.post_feedforward_layernorm.weight": shape(1)}) == "gemma4_e4b"
    assert acceleration.detect_text_encoder_family({"model.layers.34.self_attn.q_norm.weight": shape(1), "model.layers.0.post_feedforward_layernorm.weight": shape(1)}) == "gemma4_e2b"
    assert acceleration.detect_text_encoder_family({"model.layers.59.self_attn.q_norm.weight": shape(1), "model.layers.0.post_feedforward_layernorm.weight": shape(1)}) == "gemma4_31b"
    assert acceleration.detect_text_encoder_family({"model.layers.47.self_attn.q_norm.weight": shape(1), "model.layers.0.post_feedforward_layernorm.weight": shape(1)}) == "gemma3_12b"
    assert acceleration.detect_text_encoder_family({"model.language_model.layers.0.linear_attn.A_log": shape(1), "model.language_model.layers.0.input_layernorm.weight": shape(5120)}) == "qwen35_27b"


def test_detects_gemma4_assistant_config():
    assert acceleration.detect_gemma4_assistant_config({"model_type": "gemma4_assistant"}) == "gemma4_assistant"
    assert acceleration.detect_gemma4_assistant_config({"architectures": ["Gemma4AssistantForCausalLM"]}) == "gemma4_assistant"
    assert acceleration.detect_gemma4_assistant_config({"model_type": "gemma4"}) == ""


def test_gemma4_assistant_route_is_gated_until_verifier_exists(tmp_path):
    model = tmp_path / "gemma_4_E4B_it.safetensors"
    model.write_bytes(b"")
    (tmp_path / "gemma_4_E4B_it.accel.json").write_text(
        json.dumps(
            {
                "acceleration": {
                    "mode": "auto",
                    "family": "gemma4_e4b",
                    "assistant_clip_name": "gemma_4_E4B_it_assistant.safetensors",
                    "assistant_kind": "gemma4_assistant",
                    "max_speculative_tokens": 4,
                }
            }
        ),
        encoding="utf-8",
    )
    accel = acceleration.make_accelerator_for_clip(str(model), {"model.layers.41.self_attn.q_norm.weight": shape(1), "model.layers.0.post_feedforward_layernorm.weight": shape(1)})

    gate = accel.evaluate_gate(do_sample=False, tokens=[[1, 2, 3]])

    assert gate.supported is False
    assert gate.route == "torch_mps_gemma4_assistant"
    assert gate.reason == "torch_mps_gemma4_assistant_verifier_not_implemented"


def test_gemma4_assistant_hf_route_supports_existing_runtime(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(acceleration, "write_event", lambda event_type, **payload: events.append({"event_type": event_type, **payload}))
    config = acceleration.parse_acceleration_config(
        {
            "acceleration": {
                "mode": "auto",
                "family": "gemma4_e4b",
                "assistant_clip_name": "gemma_4_E4B_it_assistant.safetensors",
                "assistant_kind": "gemma4_assistant",
                "assistant_repo_id": "google/gemma-4-E4B-it-assistant",
                "max_speculative_tokens": 4,
            }
        }
    )
    accel = acceleration.TextGenerationAccelerator(config, target_clip_name="target.safetensors", target_family="gemma4_e4b", sidecar_path=str(tmp_path / "target.accel.json"))
    monkeypatch.setattr(accel, "_load_gemma4_assistant", lambda device: "assistant")
    calls = {}

    class FakeTarget:
        execution_device = "cuda"

        def generate_with_gemma4_assistant(self, tokens, **kwargs):
            calls.update(kwargs)
            assert tokens == [[1]]
            return [10, 11]

    runtime = types.SimpleNamespace(clip="gemma4", gemma4=FakeTarget())

    output = accel.generate(lambda: ["fallback"], tokens=[[1]], do_sample=False, max_length=8, sampler={"temperature": 1.0}, runtime_model=runtime)

    assert output == [10, 11]
    assert calls["assistant_model"] == "assistant"
    assert calls["max_speculative_tokens"] == 4
    assert calls["temperature"] == 0.0
    assert [event["event_type"] for event in events] == ["textgen_accel_gate", "textgen_accel_generate_end"]
    assert events[0]["supported"] is True
    assert events[0]["route"] == "torch_gemma4_assistant_hf"


def test_gemma3_is_not_treated_as_gemma4_mtp_target(tmp_path):
    config = acceleration.parse_acceleration_config(
        {
            "acceleration": {
                "mode": "auto",
                "family": "gemma3_12b",
                "assistant_clip_name": "gemma_4_E4B_it_assistant.safetensors",
                "assistant_kind": "gemma4_assistant",
            }
        }
    )
    accel = acceleration.TextGenerationAccelerator(config, target_clip_name="gemma_3_12B_it.safetensors", target_family="gemma3_12b", sidecar_path=str(tmp_path / "gemma3.accel.json"))

    gate = accel.evaluate_gate(do_sample=False, tokens=[[1, 2, 3]])

    assert gate.supported is False
    assert gate.reason == "target_draft_pair_not_verified"


def test_qwen36_native_mtp_route_is_gated_until_verifier_exists(tmp_path):
    config = acceleration.parse_acceleration_config(
        {
            "acceleration": {
                "mode": "auto",
                "family": "qwen36_next",
                "native_mtp": {"enabled": True, "max_speculative_tokens": 2},
            }
        }
    )
    accel = acceleration.TextGenerationAccelerator(config, target_clip_name="qwen3.6.safetensors", target_family="qwen36_next", sidecar_path=str(tmp_path / "qwen3.6.accel.json"))

    gate = accel.evaluate_gate(do_sample=False, tokens=[[1, 2, 3]])

    assert gate.supported is False
    assert gate.route == "torch_mps_qwen_native_mtp"
    assert gate.reason == "torch_mps_qwen_native_mtp_verifier_not_implemented"


def test_sampling_is_rejected_until_exact_speculative_sampling_exists(tmp_path):
    config = acceleration.parse_acceleration_config(
        {
            "acceleration": {
                "mode": "auto",
                "family": "gemma4_e4b",
                "assistant_clip_name": "assistant.safetensors",
                "assistant_kind": "gemma4_assistant",
            }
        }
    )
    accel = acceleration.TextGenerationAccelerator(config, target_clip_name="target.safetensors", target_family="gemma4_e4b", sidecar_path=str(tmp_path / "target.accel.json"))

    gate = accel.evaluate_gate(do_sample=True, tokens=[[1, 2, 3]])

    assert gate.supported is False
    assert gate.reason == "sampled_speculation_not_implemented"


def test_multimodal_tokens_gate_off_text_only_drafter(tmp_path):
    config = acceleration.parse_acceleration_config(
        {
            "acceleration": {
                "mode": "auto",
                "family": "gemma4_e4b",
                "assistant_clip_name": "assistant.safetensors",
                "assistant_kind": "gemma4_assistant",
                "allow_multimodal": False,
            }
        }
    )
    accel = acceleration.TextGenerationAccelerator(config, target_clip_name="target.safetensors", target_family="gemma4_e4b", sidecar_path=str(tmp_path / "target.accel.json"))

    gate = accel.evaluate_gate(do_sample=False, tokens=[[(1, 1.0), ({"type": "image", "data": object()}, 1.0)]])

    assert gate.supported is False
    assert gate.reason == "vlm_draft_not_verified"


def test_qwen_mtplx_contract_requires_runtime_and_mtp_weights(tmp_path):
    assert acceleration.qwen_mtp_contract_status(str(tmp_path)).reason == "missing_mtplx_runtime_contract"
    (tmp_path / "mtplx_runtime.json").write_text(json.dumps({"arch_id": "qwen3-next-mtp"}), encoding="utf-8")
    assert acceleration.qwen_mtp_contract_status(str(tmp_path)).reason == "missing_mtp_safetensors"
    (tmp_path / "mtp.safetensors").write_bytes(b"")

    gate = acceleration.qwen_mtp_contract_status(str(tmp_path))

    assert gate.supported is True
    assert gate.route == "mlx_mtplx_qwen"


def test_non_strict_acceleration_falls_back_to_original_generate(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(acceleration, "write_event", lambda event_type, **payload: events.append({"event_type": event_type, **payload}))
    config = acceleration.parse_acceleration_config(
        {
            "acceleration": {
                "mode": "auto",
                "family": "gemma4_e4b",
                "assistant_clip_name": "assistant.safetensors",
                "assistant_kind": "gemma4_assistant",
            }
        }
    )
    accel = acceleration.TextGenerationAccelerator(config, target_clip_name="target.safetensors", target_family="gemma4_e4b", sidecar_path=str(tmp_path / "target.accel.json"))

    output = accel.generate(lambda: [1, 2, 3], tokens=[[1]], do_sample=False, max_length=3, sampler={})

    assert output == [1, 2, 3]
    assert [event["event_type"] for event in events] == ["textgen_accel_gate", "textgen_accel_fallback"]
    assert events[0]["reason"] == "torch_mps_gemma4_assistant_verifier_not_implemented"
    assert events[1]["seconds"] >= 0
    assert events[1]["output_tokens"] == 3


def test_strict_acceleration_errors_before_generation(tmp_path):
    config = acceleration.parse_acceleration_config(
        {
            "acceleration": {
                "mode": "auto",
                "strict": True,
                "family": "gemma4_e4b",
                "assistant_clip_name": "assistant.safetensors",
                "assistant_kind": "gemma4_assistant",
            }
        }
    )
    accel = acceleration.TextGenerationAccelerator(config, target_clip_name="target.safetensors", target_family="gemma4_e4b", sidecar_path=str(tmp_path / "target.accel.json"))

    with pytest.raises(RuntimeError, match="strict but unavailable"):
        accel.generate(lambda: [1, 2, 3], tokens=[[1]], do_sample=False, max_length=3, sampler={})
