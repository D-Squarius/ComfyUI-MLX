import importlib.util
import math

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("mlx") is None, reason="mlx is not installed")


def test_ffn_silu_mul_matches_eager_mlx():
    import mlx.core as mx

    from comfy.backends.mlx_dit_native import kernels

    gate = mx.array([[-2.0, -0.5, 0.25, 3.0], [1.25, -1.5, 2.5, -3.5]], dtype=mx.float32)
    up = mx.array([[0.5, -2.0, 4.0, -1.0], [2.0, 3.0, -0.25, 0.75]], dtype=mx.float32)

    expected = (gate * mx.sigmoid(gate)) * up
    actual = kernels.ffn_silu_mul(gate, up)
    mx.eval(expected, actual)

    assert mx.max(mx.abs(expected - actual)).item() < 1.0e-6


def test_ffn_silu_mul_rejects_shape_mismatch():
    import mlx.core as mx

    from comfy.backends.mlx_dit_native import kernels

    gate = mx.zeros((1, 4), dtype=mx.float32)
    up = mx.zeros((1, 5), dtype=mx.float32)

    with pytest.raises(ValueError, match="matching shapes"):
        kernels.ffn_silu_mul(gate, up)


def test_ffn_packed_silu_mul_split_matches_eager_mlx():
    import mlx.core as mx

    from comfy.backends.mlx_dit_native import kernels

    gate = mx.array([[-2.0, -0.5, 0.25, 3.0], [1.25, -1.5, 2.5, -3.5]], dtype=mx.float32)
    up = mx.array([[0.5, -2.0, 4.0, -1.0], [2.0, 3.0, -0.25, 0.75]], dtype=mx.float32)
    packed = mx.concatenate([gate, up], axis=-1)

    expected = (gate * mx.sigmoid(gate)) * up
    actual = kernels.ffn_packed_silu_mul_split(packed)
    mx.eval(expected, actual)

    assert actual.shape == gate.shape
    assert mx.max(mx.abs(expected - actual)).item() < 1.0e-6


def test_ffn_packed_silu_mul_split_rejects_bad_hidden_dim():
    import mlx.core as mx

    from comfy.backends.mlx_dit_native import kernels

    packed = mx.zeros((1, 7), dtype=mx.float32)

    with pytest.raises(ValueError, match="even packed dim"):
        kernels.ffn_packed_silu_mul_split(packed)
    with pytest.raises(ValueError, match="2 \\* hidden_dim"):
        kernels.ffn_packed_silu_mul_split(mx.zeros((1, 8), dtype=mx.float32), hidden_dim=3)


def _rmsnorm_residual_gate_expected(mx, x, weight, residual, gate, eps=1.0e-5):
    xf = x.astype(mx.float32)
    normed = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    normed = (normed * weight.astype(mx.float32)).astype(x.dtype)
    return residual + gate * normed


@pytest.mark.parametrize(
    "gate_shape",
    [
        (4,),
        (2, 1, 4),
        (2, 3, 4),
    ],
)
def test_rmsnorm_residual_gate_matches_eager_broadcast_modes(gate_shape):
    import mlx.core as mx

    from comfy.backends.mlx_dit_native import kernels

    x = (mx.arange(2 * 3 * 4, dtype=mx.float32).reshape(2, 3, 4) - 7.0) / 11.0
    residual = (mx.arange(2 * 3 * 4, dtype=mx.float32).reshape(2, 3, 4) + 3.0) / 17.0
    weight = mx.array([0.75, 1.25, 0.5, 1.5], dtype=mx.float32)
    gate = mx.linspace(-0.5, 0.75, math.prod(gate_shape), dtype=mx.float32).reshape(*gate_shape)

    expected = _rmsnorm_residual_gate_expected(mx, x, weight, residual, gate)
    actual = kernels.rmsnorm_residual_gate(x, weight, residual, gate)
    mx.eval(expected, actual)

    assert actual.shape == x.shape
    assert mx.allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5).item()


def test_rmsnorm_residual_gate_matches_eager_bfloat16():
    import mlx.core as mx

    from comfy.backends.mlx_dit_native import kernels

    x = (mx.random.normal((1, 5, 8)) * 0.5).astype(mx.bfloat16)
    residual = (mx.random.normal((1, 5, 8)) * 0.25).astype(mx.bfloat16)
    weight = mx.linspace(0.5, 1.25, 8, dtype=mx.float32)
    gate = mx.linspace(-0.25, 0.5, 8, dtype=mx.bfloat16)

    expected = _rmsnorm_residual_gate_expected(mx, x, weight, residual, gate)
    actual = kernels.rmsnorm_residual_gate(x, weight, residual, gate)
    mx.eval(expected, actual)

    assert actual.shape == x.shape
    assert mx.all(mx.isfinite(actual)).item()
    assert mx.allclose(actual.astype(mx.float32), expected.astype(mx.float32), rtol=8.0e-3, atol=8.0e-3).item()


def test_rmsnorm_residual_gate_rejects_bad_shapes():
    import mlx.core as mx

    from comfy.backends.mlx_dit_native import kernels

    x = mx.zeros((1, 2, 4), dtype=mx.float32)
    weight = mx.ones((4,), dtype=mx.float32)
    gate = mx.ones((4,), dtype=mx.float32)

    with pytest.raises(ValueError, match="weight hidden size mismatch"):
        kernels.rmsnorm_residual_gate(x, mx.ones((5,), dtype=mx.float32), x, gate)
    with pytest.raises(ValueError, match="residual shape mismatch"):
        kernels.rmsnorm_residual_gate(x, weight, mx.zeros((1, 1, 4), dtype=mx.float32), gate)
    with pytest.raises(ValueError, match="native broadcast hidden size mismatch"):
        kernels.rmsnorm_residual_gate(x, weight, x, mx.ones((5,), dtype=mx.float32))


def _freqs(mx, batch: int, tokens: int, head_dim: int):
    half = head_dim // 2
    angle = mx.arange(batch * tokens * half, dtype=mx.float32).reshape(batch, tokens, 1, half) / 17.0
    matrix = mx.stack([mx.cos(angle), -mx.sin(angle), mx.sin(angle), mx.cos(angle)], axis=-1)
    return matrix.reshape(batch, tokens, 1, half, 2, 2)


def test_attn_prep_qknorm_rope_pack_eager_matches_z_image_reference():
    import mlx.core as mx

    from comfy.backends.mlx_dit_native import kernels
    from comfy.backends.mlx_z_image import _apply_rope_head_major, _rms_norm

    q = mx.arange(2 * 3 * 2 * 4, dtype=mx.float32).reshape(2, 3, 2, 4) / 23.0
    k = (mx.arange(2 * 3 * 2 * 4, dtype=mx.float32).reshape(2, 3, 2, 4) - 7.0) / 19.0
    v = (mx.arange(2 * 3 * 2 * 4, dtype=mx.float32).reshape(2, 3, 2, 4) + 5.0) / 29.0
    q_weight = mx.array([0.75, 1.25, 0.5, 1.5], dtype=mx.float32)
    k_weight = mx.array([1.5, 0.5, 1.25, 0.75], dtype=mx.float32)
    freqs = _freqs(mx, 2, 3, 4)

    expected_q = _apply_rope_head_major(mx, _rms_norm(mx, q.transpose(0, 2, 1, 3), q_weight), freqs)
    expected_k = _apply_rope_head_major(mx, _rms_norm(mx, k.transpose(0, 2, 1, 3), k_weight), freqs)
    expected_v = v.transpose(0, 2, 1, 3)
    actual_q, actual_k, actual_v = kernels.attn_prep_qknorm_rope_pack_eager(mx, q, k, v, q_weight, k_weight, freqs)
    mx.eval(expected_q, expected_k, expected_v, actual_q, actual_k, actual_v)

    assert mx.allclose(actual_q, expected_q, rtol=1.0e-5, atol=1.0e-5).item()
    assert mx.allclose(actual_k, expected_k, rtol=1.0e-5, atol=1.0e-5).item()
    assert mx.allclose(actual_v, expected_v, rtol=1.0e-6, atol=1.0e-6).item()


def test_attn_prep_qknorm_rope_pack_native_matches_eager():
    import mlx.core as mx

    from comfy.backends.mlx_dit_native import kernels

    q = mx.random.normal((1, 5, 3, 8)).astype(mx.float32)
    k = mx.random.normal((1, 5, 3, 8)).astype(mx.float32)
    v = mx.random.normal((1, 5, 3, 8)).astype(mx.float32)
    q_weight = mx.linspace(0.75, 1.25, 8, dtype=mx.float32)
    k_weight = mx.linspace(1.25, 0.75, 8, dtype=mx.float32)
    freqs = _freqs(mx, 1, 5, 8)

    expected = kernels.attn_prep_qknorm_rope_pack_eager(mx, q, k, v, q_weight, k_weight, freqs)
    actual = kernels.attn_prep_qknorm_rope_pack(q, k, v, q_weight, k_weight, freqs)
    mx.eval(*expected, *actual)

    for expected_part, actual_part in zip(expected, actual):
        assert mx.allclose(actual_part, expected_part, rtol=1.0e-5, atol=1.0e-5).item()


def test_attn_prep_qknorm_rope_pack_rejects_shape_mismatch():
    import mlx.core as mx

    from comfy.backends.mlx_dit_native import kernels

    q = mx.zeros((1, 2, 1, 4), dtype=mx.float32)
    k = mx.zeros((1, 2, 1, 5), dtype=mx.float32)
    v = mx.zeros((1, 2, 1, 4), dtype=mx.float32)
    w = mx.ones((4,), dtype=mx.float32)
    freqs = _freqs(mx, 1, 2, 4)

    with pytest.raises(ValueError, match="matching q/k/v"):
        kernels.attn_prep_qknorm_rope_pack(q, k, v, w, w, freqs)


def test_native_bf16_self_attention_matches_mlx_reference_small_shape():
    import mlx.core as mx

    from comfy.backends.mlx_dit_native import kernels

    q = (mx.arange(1 * 4 * 2 * 4, dtype=mx.float32).reshape(1, 4, 2, 4) - 3.0) / 17.0
    k = (mx.arange(1 * 4 * 2 * 4, dtype=mx.float32).reshape(1, 4, 2, 4) + 5.0) / 19.0
    v = (mx.arange(1 * 4 * 2 * 4, dtype=mx.float32).reshape(1, 4, 2, 4) - 7.0) / 23.0
    q_weight = mx.linspace(0.75, 1.25, 4, dtype=mx.float32)
    k_weight = mx.linspace(1.25, 0.75, 4, dtype=mx.float32)
    freqs = _freqs(mx, 1, 4, 4)

    expected = kernels.native_bf16_self_attention_reference(mx, q, k, v, q_weight, k_weight, freqs, scale=0.5)
    actual = kernels.native_bf16_self_attention(q, k, v, q_weight, k_weight, freqs, scale=0.5)
    mx.eval(expected, actual)

    assert actual.shape == q.shape
    assert mx.allclose(actual, expected, rtol=2.0e-4, atol=2.0e-4).item()


def test_native_bf16_self_attention_rejects_large_token_count(monkeypatch):
    import mlx.core as mx

    from comfy.backends.mlx_dit_native import kernels

    monkeypatch.setenv("COMFY_MLX_DIT_NATIVE_ATTENTION_MAX_TOKENS", "2")
    q = mx.zeros((1, 3, 1, 4), dtype=mx.float32)
    w = mx.ones((4,), dtype=mx.float32)
    freqs = _freqs(mx, 1, 3, 4)

    with pytest.raises(ValueError, match="disabled for large token counts"):
        kernels.native_bf16_self_attention(q, q, q, w, w, freqs, scale=0.5)
