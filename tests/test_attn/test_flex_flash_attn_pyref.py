# Copyright (c) 2025 SandAI.
# Test FFA Python reference forward against PyTorch attention reference.

import math
import os
import random
import sys
import unittest

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.common import AttnRanges
from magi_attention.utils import get_attn_mask_from_ffa_args
from magi_attention.testing.precision import (
    EPSILON,
    assert_close,
    torch_attn_ref,
)
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp

try:
    from tests._pyref.ffa_pyop import flex_flash_attn_pyref
except ImportError as e:
    print(f"Warning: Failed to import flex_flash_attn_pyref: {e}")
    print(f"Python path: {sys.path}")
    raise


class TestFlexFlashAttnPyRef(DistTestBase):
    @property
    def seed(self):
        return 42

    @property
    def device(self):
        return torch.cuda.current_device()

    MODEL_CONFIGS = [
        {"name": "mha_nh8_hd128", "num_heads_q": 8, "num_heads_kv": 8, "head_dim": 128},
        {"name": "gqa_nhq32_nhkv1_hd128", "num_heads_q": 32, "num_heads_kv": 1, "head_dim": 128},
        {"name": "mha_nh1_hd64", "num_heads_q": 1, "num_heads_kv": 1, "head_dim": 64},
        {"name": "gqa_nhq4_nhkv2_hd64", "num_heads_q": 4, "num_heads_kv": 2, "head_dim": 64},
        {"name": "gqa_nhq8_nhkv2_hd128", "num_heads_q": 8, "num_heads_kv": 2, "head_dim": 128},
        {"name": "gqa_nhq16_nhkv4_hd64", "num_heads_q": 16, "num_heads_kv": 4, "head_dim": 64},
        {"name": "mha_nh4_hd256", "num_heads_q": 4, "num_heads_kv": 4, "head_dim": 256},
        {"name": "mha_nh16_hd128", "num_heads_q": 16, "num_heads_kv": 16, "head_dim": 128},
    ]

    ATTENTION_MASK_CONFIGS = [
        {
            "name": "full_4k",
            "seqlen": 4096,
            "q_ranges": AttnRanges.from_ranges([[0, 4096]]),
            "k_ranges": AttnRanges.from_ranges([[0, 4096]]),
            "attn_type_map": [0],
        },
        {
            "name": "varlen_full_1k",
            "seqlen": 1024,
            "q_ranges": AttnRanges.from_ranges(
                [[0, 366],[366, 391],[391, 471],[471, 835],[835, 984],[984, 1005],[1005, 1017],[1017, 1020],[1020, 1023],[1023, 1024]]
            ),
            "k_ranges": AttnRanges.from_ranges(
                [[0, 366],[366, 391],[391, 471],[471, 835],[835, 984],[984, 1005],[1005, 1017],[1017, 1020],[1020, 1023],[1023, 1024]]
            ),
            "attn_type_map": [0] * 10,
        },
        {
            "name": "varlen_block_causal_2k",
            "seqlen": 2048,
            "q_ranges": AttnRanges.from_ranges([[0,256],[256,512],[512,1024],[1024,1280],[1280,1536],[1536,1792],[1792,2048]]),
            "k_ranges": AttnRanges.from_ranges([[0,256],[0,512],[0,1024],[1024,1280],[1024,1536],[1024,1792],[1024,2048]]),
            "attn_type_map": [0] * 7,
        },
        {
            "name": "sparse_attn_2k",
            "seqlen": 2048,
            "q_ranges": AttnRanges.from_ranges(
                [[0,256],[0,256],[0,256],[256,512],[256,512],[512,1024],[1024,1280],[1280,1536],[1280,1536],[1280,1536],[1280,1536],[1280,1536],[1536,1792],[1792,2048]]
            ),
            "k_ranges": AttnRanges.from_ranges(
                [[0,256],[512,768],[1011,1123],[0,512],[777,888],[0,1024],[1024,1280],[0,128],[555,556],[777,982],[1024,1536],[1689,1898],[1024,1792],[1024,2048]]
            ),
            "attn_type_map": [0] * 14,
        },
        {
            "name": "causal_512",
            "seqlen": 512,
            "q_ranges": AttnRanges.from_ranges([[0, 512]]),
            "k_ranges": AttnRanges.from_ranges([[0, 512]]),
            "attn_type_map": [1],
        },
        {
            "name": "causal_2k",
            "seqlen": 2048,
            "q_ranges": AttnRanges.from_ranges([[0, 2048]]),
            "k_ranges": AttnRanges.from_ranges([[0, 2048]]),
            "attn_type_map": [1],
        },
        {
            "name": "varlen_causal_1k",
            "seqlen": 1024,
            "q_ranges": AttnRanges.from_ranges([[0, 256], [256, 512], [512, 768], [768, 1024]]),
            "k_ranges": AttnRanges.from_ranges([[0, 256], [256, 512], [512, 768], [768, 1024]]),
            "attn_type_map": [1, 1, 1, 1],
        },
        {
            "name": "block_diagonal_1k",
            "seqlen": 1024,
            "q_ranges": AttnRanges.from_ranges([[0, 256], [256, 512], [512, 768], [768, 1024]]),
            "k_ranges": AttnRanges.from_ranges([[0, 256], [256, 512], [512, 768], [768, 1024]]),
            "attn_type_map": [0, 0, 0, 0],
        },
        {
            "name": "tiny_seq_64",
            "seqlen": 64,
            "q_ranges": AttnRanges.from_ranges([[0, 64]]),
            "k_ranges": AttnRanges.from_ranges([[0, 64]]),
            "attn_type_map": [0],
        },
        {
            "name": "tiny_varlen_128",
            "seqlen": 128,
            "q_ranges": AttnRanges.from_ranges([[0, 32], [32, 64], [64, 96], [96, 128]]),
            "k_ranges": AttnRanges.from_ranges([[0, 32], [32, 64], [64, 96], [96, 128]]),
            "attn_type_map": [0, 0, 0, 0],
        },
        {
            "name": "mixed_types_1k",
            "seqlen": 1024,
            "q_ranges": AttnRanges.from_ranges([[0, 256], [256, 512], [512, 768], [768, 1024]]),
            "k_ranges": AttnRanges.from_ranges([[0, 256], [256, 512], [512, 768], [768, 1024]]),
            "attn_type_map": [0, 1, 0, 1],
        },
    ]

    def _assert_close_to_torch(self, q, k, v, q_ranges, k_ranges, attn_type_map, o, dtype, test_case="", softcap=0.0):
        """Compare output with PyTorch reference implementation."""
        try:
            mask = get_attn_mask_from_ffa_args(
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
                total_seqlen_q=q.shape[0],
                total_seqlen_k=k.shape[0],
                device=q.device,
            )
            o_ref = torch_attn_ref(q, k, v, mask=mask, layout="thd", high_precision=True, softcap=softcap)

            # PyRef uses float32 intermediate computation but outputs fp16/bf16
            # Requires relaxed tolerance considering sequence length and softmax accumulation error
            o_atol = {torch.bfloat16: 2e-3, torch.float16: 1e-3}.get(dtype, 1e-4)
            o_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)

            assert_close(o, o_ref, atol=o_atol, rtol=o_rtol, mismatch_threshold=0.02, test_case=f"{test_case} => o")
        except Exception as e:
            print(f"Error in {test_case}: {e}")
            raise

    @with_run_in_mp
    @parameterize("attn_mask_config", ATTENTION_MASK_CONFIGS)
    @parameterize("model_config", MODEL_CONFIGS)
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    @parameterize("random_attn_type_map", [False, True])
    def test_pyref_against_torch(
        self,
        attn_mask_config: dict,
        model_config: dict,
        dtype: torch.dtype,
        random_attn_type_map: bool,
    ):
        """Test PyRef forward pass against PyTorch reference."""
        seqlen = attn_mask_config["seqlen"]
        q_ranges: AttnRanges = attn_mask_config["q_ranges"]
        k_ranges: AttnRanges = attn_mask_config["k_ranges"]
        attn_type_map: list[int] = attn_mask_config["attn_type_map"]
        if random_attn_type_map:
            attn_type_map = torch.randint(0, 4, (len(attn_type_map),), device="cpu").tolist()

        Hq, Hk, D = model_config["num_heads_q"], model_config["num_heads_kv"], model_config["head_dim"]
        q = torch.randn(seqlen, Hq, D, dtype=dtype, device="cuda", requires_grad=False)
        k = torch.randn(seqlen, Hk, D, dtype=dtype, device="cuda", requires_grad=False)
        v = torch.randn(seqlen, Hk, D, dtype=dtype, device="cuda", requires_grad=False)

        try:
            o, lse = flex_flash_attn_pyref(
                q=q, k=k, v=v,
                q_ranges=q_ranges, k_ranges=k_ranges, attn_type_map=attn_type_map,
                softmax_scale=None, softcap=0.0, sm_margin=0, out_type=None, deterministic=False, auto_range_merge=False
            )
        except Exception as e:
            print(f"Error in flex_flash_attn_pyref: {e}")
            raise

        self._assert_close_to_torch(
            q, k, v, q_ranges, k_ranges, attn_type_map, o, dtype,
            test_case=f"[{attn_mask_config['name']}][{model_config['name']}][{dtype=}]"
        )

    @with_run_in_mp
    @parameterize("seqlen", [256, 512, 1024, 2048])
    @parameterize("num_heads_q", [1, 4, 8])
    @parameterize("num_heads_kv", [1, 2])
    @parameterize("head_dim", [64, 128])
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    @parameterize("softcap", [0.0, 30.0, 50.0])
    def test_pyref_with_softcap(
        self,
        seqlen: int,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        dtype: torch.dtype,
        softcap: float,
    ):
        """Test softcap parameter."""
        q_ranges = AttnRanges.from_ranges([[0, seqlen]])
        k_ranges = AttnRanges.from_ranges([[0, seqlen]])
        attn_type_map = [0]

        q = torch.randn(seqlen, num_heads_q, head_dim, dtype=dtype, device="cuda", requires_grad=False)
        k = torch.randn(seqlen, num_heads_kv, head_dim, dtype=dtype, device="cuda", requires_grad=False)
        v = torch.randn(seqlen, num_heads_kv, head_dim, dtype=dtype, device="cuda", requires_grad=False)

        o, lse = flex_flash_attn_pyref(
            q=q, k=k, v=v,
            q_ranges=q_ranges, k_ranges=k_ranges, attn_type_map=attn_type_map,
            softmax_scale=None, softcap=softcap, sm_margin=0, out_type=None, deterministic=False, auto_range_merge=False
        )

        self._assert_close_to_torch(
            q, k, v, q_ranges, k_ranges, attn_type_map, o, dtype,
            test_case=f"[softcap={softcap}][{seqlen=}][nhq={num_heads_q}][nhk={num_heads_kv}][{head_dim=}][{dtype=}]",
            softcap=softcap
        )

    @with_run_in_mp
    @parameterize("seqlen", [512, 1024])
    @parameterize("softmax_scale", [None, 0.5, 1.0, 2.0])
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    def test_pyref_custom_softmax_scale(
        self,
        seqlen: int,
        softmax_scale: float,
        dtype: torch.dtype,
    ):
        """Test custom softmax_scale parameter."""
        num_heads_q, num_heads_kv, head_dim = 8, 8, 128
        q_ranges = AttnRanges.from_ranges([[0, seqlen]])
        k_ranges = AttnRanges.from_ranges([[0, seqlen]])
        attn_type_map = [0]

        q = torch.randn(seqlen, num_heads_q, head_dim, dtype=dtype, device="cuda", requires_grad=False)
        k = torch.randn(seqlen, num_heads_kv, head_dim, dtype=dtype, device="cuda", requires_grad=False)
        v = torch.randn(seqlen, num_heads_kv, head_dim, dtype=dtype, device="cuda", requires_grad=False)

        o, lse = flex_flash_attn_pyref(
            q=q, k=k, v=v,
            q_ranges=q_ranges, k_ranges=k_ranges, attn_type_map=attn_type_map,
            softmax_scale=softmax_scale, softcap=0.0, sm_margin=0, out_type=None, deterministic=False, auto_range_merge=False
        )

        # Generate torch reference with same scale
        mask = get_attn_mask_from_ffa_args(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=q.shape[0],
            total_seqlen_k=k.shape[0],
            device=q.device,
        )
        scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(head_dim))
        o_ref = torch_attn_ref(q, k, v, mask=mask, layout="thd", high_precision=True, scale=scale)

        o_atol = {torch.bfloat16: 2e-3, torch.float16: 1e-3}.get(dtype, 1e-4)
        o_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)

        assert_close(o, o_ref, atol=o_atol, rtol=o_rtol, mismatch_threshold=0.02,
                    test_case=f"[softmax_scale={softmax_scale}][{seqlen=}][{dtype=}] => o")

    @with_run_in_mp
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    def test_pyref_edge_cases(self, dtype: torch.dtype):
        """Test edge cases with minimal sequence length."""
        seqlen = 1
        num_heads_q, num_heads_kv, head_dim = 1, 1, 64
        q_ranges = AttnRanges.from_ranges([[0, seqlen]])
        k_ranges = AttnRanges.from_ranges([[0, seqlen]])
        attn_type_map = [0]

        q = torch.randn(seqlen, num_heads_q, head_dim, dtype=dtype, device="cuda", requires_grad=False)
        k = torch.randn(seqlen, num_heads_kv, head_dim, dtype=dtype, device="cuda", requires_grad=False)
        v = torch.randn(seqlen, num_heads_kv, head_dim, dtype=dtype, device="cuda", requires_grad=False)

        o, lse = flex_flash_attn_pyref(
            q=q, k=k, v=v,
            q_ranges=q_ranges, k_ranges=k_ranges, attn_type_map=attn_type_map,
            softmax_scale=None, softcap=0.0, sm_margin=0, out_type=None, deterministic=False, auto_range_merge=False
        )

        self._assert_close_to_torch(
            q, k, v, q_ranges, k_ranges, attn_type_map, o, dtype,
            test_case=f"[edge_case_seqlen_1][{dtype=}]"
        )


if __name__ == "__main__":
    run_tests()