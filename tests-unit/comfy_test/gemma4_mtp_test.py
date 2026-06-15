import torch

import comfy.ops
from comfy.text_encoders.gemma4 import Gemma4Base, Gemma4Config, Gemma4Transformer


class _MinimalGemma4(Gemma4Base):
    pass


class _FakeAssistantOutput:
    def __init__(self, logits, last_hidden_state):
        self.logits = logits
        self.last_hidden_state = last_hidden_state


class _FakeAssistant(torch.nn.Module):
    def __init__(self, vocab_size, token_id, hidden_size):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size
        self.token_id = token_id
        self.hidden_size = hidden_size

    def forward(self, inputs_embeds, **kwargs):
        logits = torch.zeros(inputs_embeds.shape[0], inputs_embeds.shape[1], self.vocab_size, device=inputs_embeds.device)
        logits[..., self.token_id] = 1.0
        return _FakeAssistantOutput(logits, inputs_embeds[..., : self.hidden_size])


def _tiny_target():
    cfg = Gemma4Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=1,
        final_norm=True,
    )
    cfg.head_dim = 4
    cfg.global_head_dim = 4
    cfg.final_logit_softcapping = 0
    cfg.sliding_attention = [False]
    cfg.stop_tokens = [31]
    target = _MinimalGemma4()
    target.model = Gemma4Transformer(cfg, device="cpu", dtype=torch.float32, ops=comfy.ops.manual_cast)
    target.num_layers = cfg.num_hidden_layers
    target.dtype = torch.float32
    return target


def test_gemma4_assistant_loop_verifies_multi_token_drafts_with_past_cache():
    target = _tiny_target()
    input_ids = torch.tensor([[1, 2, 3]])
    embeds = target.model.embed_tokens(input_ids, out_dtype=torch.float32)

    output = target.generate_with_assistant(
        embeds,
        input_ids,
        _FakeAssistant(vocab_size=32, token_id=2, hidden_size=8),
        max_speculative_tokens=2,
        max_length=3,
    )

    assert len(output) == 3
    assert all(isinstance(token, int) for token in output)


def test_gemma4_alternative_attention_uses_global_kv_heads_without_v_proj():
    cfg = Gemma4Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        final_norm=True,
    )
    cfg.head_dim = 4
    cfg.global_head_dim = 4
    cfg.final_logit_softcapping = 0
    cfg.sliding_attention = [False]
    target = Gemma4Transformer(cfg, device="cpu", dtype=torch.float32, ops=comfy.ops.manual_cast)

    attn = target.layers[0].self_attn
    assert attn.v_proj is None
    assert attn.num_kv_heads == 1

    input_ids = torch.tensor([[1, 2]])
    output, _, past = target(input_ids, dtype=torch.float32, past_key_values=[()])

    assert output.shape == (1, 2, 8)
    assert len(past) == 1
