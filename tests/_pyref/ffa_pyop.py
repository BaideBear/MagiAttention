"""
Pure-Python reference for Flexible Flash Attention (forward only).
- Mimics real operator execution: block partition, special masks, block range calc.
- Ignores FF3-specific traits (ping-pong schedule, producer-consumer, etc.).
- Supports MHA and GQA: num_heads_q may differ from num_heads_kv (use h % Hkv).

Returns out in q.dtype by caller conversion; accumulations/lse are fp32.
"""
import math
import torch
from typing import Tuple, Optional
from magi_attention.utils import get_attn_mask_from_ffa_args


def _get_block_sizes(head_dim: int) -> Tuple[int, int]:
    # Heuristic similar to tile_size_fwd_sm90
    if head_dim <= 64:
        return 192, 128
    elif head_dim <= 128:
        return 64, 64
    elif head_dim <= 192:
        return 128, 96
    else:
        return 128, 64


essential_masks = {
    0: "full",
    1: "causal",
    2: "invcausal",
    3: "bicausal",
}


def _n_block_min_max(m_block: int, kBlockM: int, kBlockN: int, seqlen_q: int, seqlen_k: int, attn_type: int) -> Tuple[int, int]:
    n_max = math.ceil(seqlen_k / kBlockN)
    n_min = 0

    if attn_type == 0:  # full
        return n_min, n_max

    if attn_type in (1, 3):  # causal
        m_idx_max = min(seqlen_q, (m_block + 1) * kBlockM)
        causal_limit = max(0, m_idx_max + seqlen_k - seqlen_q)
        n_max = min(n_max, math.ceil(causal_limit / kBlockN))

    if attn_type in (2, 3):  # invcausal
        m_idx_min = m_block * kBlockM
        if m_idx_min >= seqlen_k:
            n_min = n_max
        else:
            # InvCausal: row i attends to column j where j >= i
            # Minimum valid K block: first block containing columns >= m_idx_min
            n_min = m_idx_min // kBlockN

    return n_min, n_max


def _apply_mask_block(scores_2d: torch.Tensor, m_block: int, n_block: int, kBlockM: int, kBlockN: int, seqlen_q: int, seqlen_k: int, attn_type: int) -> None:
    # scores_2d: [bm, bn] local to current q/k ranges
    bm, bn = scores_2d.shape
    device = scores_2d.device

    row_start = m_block * kBlockM
    col_start = n_block * kBlockN
    rows = torch.arange(bm, device=device).unsqueeze(1) + row_start
    cols = torch.arange(bn, device=device).unsqueeze(0) + col_start

    valid = torch.ones((bm, bn), dtype=torch.bool, device=device)
    valid &= rows < seqlen_q
    valid &= cols < seqlen_k

    if attn_type in (1, 3):  # causal (bottom-right aligned)
        # q_i attends to k_j if j < i + (seqlen_k - seqlen_q) + 1
        col_limit = rows + (seqlen_k - seqlen_q) + 1
        valid &= cols < col_limit

    if attn_type in (2, 3):  # inverse causal (top-left aligned)
        # q_i attends to k_j if j >= i - (seqlen_k - seqlen_q) = i + (seqlen_q - seqlen_k)
        col_lower = rows + (seqlen_q - seqlen_k)
        valid &= cols >= col_lower

    scores_2d.masked_fill_(~valid, float("-inf"))


def flex_flash_attn_pyref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges,          # AttnRanges (推荐) 或 tensor(形如 [N,2])
    k_ranges,          # 同上
    attn_type_map,     # list[int] 或 int tensor
    *,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    sm_margin: int = 0,
    out_type: torch.dtype | None = None,
    deterministic: bool = False,
    auto_range_merge: bool = False,
):
    """
    PyRef 等价实现：
    - 使用 get_attn_mask_from_ffa_args 严格对齐 mask 语义（包括非方形、稀疏、attn_type_map 含义）。
    - 支持 GQA：当 Hq != Hk 时将 K/V 沿 head 维广播/重复至 Hq。
    - logits 使用 masked_fill(-inf) 再 softmax；lse 为 logsumexp。
    - 返回 (o, lse)，其中 o dtype=out_type(若设置)或 q.dtype，lse=float32。
    """
    # shapes: q=[Tq, Hq, D], k=[Tk, Hk, D], v=[Tk, Hk, D]
    Tq, Hq, D = q.shape
    Tk, Hk, Dk = k.shape
    assert D == Dk == v.shape[-1], "Q/K/V head_dim 不一致"
    assert v.shape[:2] == (Tk, Hk), "V 的前两维必须与 K 一致"

    # softmax scale
    if softmax_scale is None:
        softmax_scale = 1.0 / (D ** 0.5)

    # 生成全局 [Tq, Tk] 掩码，确保与 C++/CUDA 实现的语义一致
    # 建议传入 AttnRanges，若已是 tensor 则调用方在测试中传 AttnRanges 即可（推荐方式）
    mask = get_attn_mask_from_ffa_args(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map if isinstance(attn_type_map, list) else attn_type_map.tolist(),
        total_seqlen_q=Tq,
        total_seqlen_k=Tk,
        device=q.device,
    )  # bool, shape=[Tq, Tk]

    # 处理 GQA：将 K/V 的 head 维广播至 Hq
    if Hq != Hk:
        assert Hq % Hk == 0, "GQA 条件不满足：Hq 必须是 Hk 的整数倍"
        repeat = Hq // Hk
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    # 变换为 [H, T, D] 便于批量 matmul
    q_h = q.permute(1, 0, 2).contiguous()   # [Hq, Tq, D]
    k_h = k.permute(1, 0, 2).contiguous()   # [Hq, Tk, D]
    v_h = v.permute(1, 0, 2).contiguous()   # [Hq, Tk, D]

    # logits: [H, Tq, Tk]
    logits = torch.matmul(q_h, k_h.transpose(-1, -2)) * softmax_scale

    # softcap（当前测试均为0，不做改动；保留参数占位）
    if softcap and softcap > 0:
        # 简单截断实现；如需严格对齐可替换为与内核一致的"软上限"策略
        logits = torch.clamp(logits, max=softcap)

    # 扩展 mask 至 [H, Tq, Tk]
    mask_ = mask.unsqueeze(0)  # [1, Tq, Tk]
    logits = logits.masked_fill(~mask_, float("-inf"))

    # softmax + lse（float32 以保证稳定性）
    logits_fp32 = logits.to(torch.float32)
    lse = torch.logsumexp(logits_fp32, dim=-1)  # [H, Tq]
    probs = torch.softmax(logits_fp32, dim=-1)  # [H, Tq, Tk]

    # 计算输出（将 v_h 也转为 float32 以避免类型不匹配）
    out = torch.matmul(probs, v_h.to(torch.float32))  # [H, Tq, D]
    out = out.permute(1, 0, 2).contiguous()  # [Tq, Hq, D]

    if out_type is not None:
        out = out.to(out_type)
    else:
        out = out.to(q.dtype)

    # lse 返回 float32
    lse = lse.permute(1, 0).contiguous()  # [Tq, Hq]
    return out, lse


def ffa_forward_py(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor,
    softmax_scale: Optional[float] = None,
    ref_block_size: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass (fp32 accumulation) that mirrors FFA execution at algorithmic level.
    Shapes:
      q: [Tq, Hq, D]
      k: [Tk, Hkv, D]
      v: [Tk, Hkv, D]
      q_ranges, k_ranges: [P, 2]
      attn_type_map: [P]
    Return:
      out: [Tq, Hq, D] (fp32)
      lse: [Tq, Hq] (fp32)
    """
    Tq, Hq, D = q.shape
    Tk, Hkv, Dk = k.shape
    assert Dk == D and v.shape[-1] == D

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    if ref_block_size is not None:
        kBlockM, kBlockN = ref_block_size
    else:
        kBlockM, kBlockN = _get_block_sizes(D)

    out = torch.zeros((Tq, Hq, D), dtype=torch.float32, device=q.device)
    lse = torch.full((Tq, Hq), float("-inf"), dtype=torch.float32, device=q.device)

    P = q_ranges.shape[0]
    for p in range(P):
        q_start, q_end = q_ranges[p].tolist()
        k_start, k_end = k_ranges[p].tolist()
        attn_type = int(attn_type_map[p].item()) if attn_type_map is not None else 0
        Sq = q_end - q_start
        Sk = k_end - k_start
        if Sq <= 0 or Sk <= 0:
            continue

        q_curr = q[q_start:q_end]
        k_curr = k[k_start:k_end]
        v_curr = v[k_start:k_end]

        num_m_blocks = math.ceil(Sq / kBlockM)
        for m_block in range(num_m_blocks):
            m0 = m_block * kBlockM
            m1 = min(m0 + kBlockM, Sq)
            bm = m1 - m0

            q_blk = q_curr[m0:m1]  # [bm, Hq, D]
            m_i = torch.full((bm, Hq), float("-inf"), dtype=torch.float32, device=q.device)
            l_i = torch.zeros((bm, Hq), dtype=torch.float32, device=q.device)
            O_i = torch.zeros((bm, Hq, D), dtype=torch.float32, device=q.device)

            n_min, n_max = _n_block_min_max(m_block, kBlockM, kBlockN, Sq, Sk, attn_type)
            for n_block in range(n_min, n_max):
                n0 = n_block * kBlockN
                n1 = min(n0 + kBlockN, Sk)
                bn = n1 - n0

                k_blk = k_curr[n0:n1]  # [bn, Hkv, D]
                v_blk = v_curr[n0:n1]  # [bn, Hkv, D]

                scores = torch.empty((bm, Hq, bn), dtype=torch.float32, device=q.device)
                for h in range(Hq):
                    h_kv = h % Hkv
                    scores[:, h, :] = torch.matmul(
                        q_blk[:, h, :].float(), k_blk[:, h_kv, :].float().transpose(0, 1)
                    ) * softmax_scale

                for h in range(Hq):
                    _apply_mask_block(scores[:, h, :], m_block, n_block, kBlockM, kBlockN, Sq, Sk, attn_type)

                s_max = scores.max(dim=2).values
                m_new = torch.maximum(m_i, s_max)
                
                # For numerical stability, handle -inf carefully:
                # When scores[i] == -inf, exp(scores[i] - m_new[i]) should be 0, not 1
                # This happens when m_i[i] == -inf and s_max[i] == -inf
                valid_scores = ~torch.isinf(scores)
                exp_scores = torch.zeros_like(scores)
                m_new_expanded = m_new.unsqueeze(2)
                exp_scores[valid_scores] = torch.exp((scores - m_new_expanded)[valid_scores])
                
                corr = torch.exp(m_i - m_new)
                l_i = l_i * corr + exp_scores.sum(dim=2)

                for h in range(Hq):
                    h_kv = h % Hkv
                    O_i[:, h, :] = O_i[:, h, :] * corr[:, h].unsqueeze(1)
                    O_i[:, h, :] += torch.matmul(exp_scores[:, h, :], v_blk[:, h_kv, :].float())

                m_i = m_new

            # Handle case where some rows have no valid attention (l_i == 0)
            O_block = torch.where(
                (l_i > 0).unsqueeze(2),
                O_i / (l_i.unsqueeze(2) + 1e-10),
                torch.zeros_like(O_i)
            )
            lse_block = torch.where(
                l_i > 0,
                torch.log(l_i + 1e-10) + m_i,
                torch.full_like(m_i, float("-inf"))
            )

            out[q_start + m0 : q_start + m1] = O_block
            lse[q_start + m0 : q_start + m1] = lse_block

    return out, lse
