"""Tiny stand-in for Audio8's ArkttsModel with the same interfaces the notebook uses."""
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel

SEM = 256
V = SEM + 4096


class FakeConfig(PretrainedConfig):
    model_type = "fakearktts"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.num_codebooks = 10
        self.semantic_begin_id = SEM
        self.semantic_end_id = SEM + 4095
        self.pad_token_id = 0
        self.max_seq_len = 2048
        self.norm_fastlayer_input = True
        self.codec_sample_rate = 44100
        self.use_gradient_checkpointing = False


class Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.kv_cache = None

    def forward(self, x, mask):
        if self.kv_cache is not None:
            raise RuntimeError("kv_cache left over from generate() during training!")
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attention = Attn(d)
        self.mlp = nn.Linear(d, d)

    def forward(self, x, rope, attention_mask, cache_position=None):
        return x + torch.tanh(self.mlp(self.attention(x, attention_mask)))


class FakeModel(PreTrainedModel):
    config_class = FakeConfig
    base_model_prefix = ""

    def __init__(self, config):
        super().__init__(config)
        d = 32
        self.embeddings = nn.Embedding(V, d)
        self.codebook_embeddings = nn.Embedding(10 * 4096, d)
        self.layers = nn.ModuleList([Block(d) for _ in range(2)])
        self.norm = nn.LayerNorm(d)
        self.fast_project_in = nn.Linear(d, d)
        self.fast_embeddings = nn.Embedding(4096, d)
        self.fast_layers = nn.ModuleList([Block(d)])
        self.fast_norm = nn.LayerNorm(d)
        self.fast_output = nn.Linear(d, 4096)
        self.register_buffer("fast_freqs_cis", torch.zeros(16, 4), persistent=False)
        self.post_init()

    def _causal_mask(self, attention_mask, positions, length):
        return torch.ones(attention_mask.shape[0], 1, length, length, dtype=torch.bool, device=attention_mask.device)

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=None, **kwargs):
        ids = input_ids[:, 0]
        hidden = self.embeddings(ids)
        offsets = torch.arange(10, device=ids.device)[None, :, None] * 4096
        cb = self.codebook_embeddings(input_ids[:, 1:] + offsets).sum(1)
        sem = (ids >= SEM).unsqueeze(-1)
        hidden = hidden + cb * sem
        for layer in self.layers:
            hidden = layer(hidden, None, attention_mask)
        logits = F.linear(self.norm(hidden), self.embeddings.weight)
        return SimpleNamespace(logits=logits, hidden_states=hidden)

    @torch.inference_mode()
    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=10, temperature=None, top_p=None,
                 top_k=None, do_sample=None, return_dict_in_generate=False, generator=None, **kwargs):
        for layer in list(self.layers) + list(self.fast_layers):
            layer.attention.kv_cache = nn.Linear(2, 2)  # mimic static caches being attached
        T = min(int(max_new_tokens), 40)
        codes = torch.randint(0, 4096, (1, 10, T), generator=generator, device=self.device)
        return SimpleNamespace(codes=codes, finished=torch.tensor([True]), code_lengths=torch.tensor([T]))

    def decode_audio(self, codes):
        n = codes.shape[-1] * 2048
        return torch.zeros(codes.shape[0], n) + 0.01, torch.tensor([n])

    def load_codec(self, device=None, dtype=None):
        return None

    def encode_audio(self, values, lengths):
        frames = (lengths + 2047) // 2048
        codes = torch.randint(0, 4096, (values.shape[0], 10, int(frames.max())))
        return codes, frames


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __call__(self, text, reference_audio=None, reference_text=None, return_tensors="pt"):
        ids = torch.tensor([list(text[0].encode("utf-8"))[:50]])
        out = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if reference_audio:
            out["reference_audio_values"] = torch.zeros(1, 1, 4410)
        return out
