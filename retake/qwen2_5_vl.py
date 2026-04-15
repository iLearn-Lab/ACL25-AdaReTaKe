import math
import os
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
import numpy as np

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    logging,
)
from transformers import Qwen2VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    repeat_kv,
    apply_multimodal_rotary_pos_emb,
)

from .visual_compression import *
from .longvideo_cache import *

DEBUG_MODE = False

logger = logging.get_logger(__name__)


def retake_Qwen2_5_VLAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs,  # absorb extra kwargs from transformers >= 4.57 (FlashAttentionKwargs etc.)
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # transformers >= 4.57 renames past_key_value -> past_key_values
    if past_key_value is None:
        past_key_value = kwargs.pop('past_key_values', None)
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    # Update position_ids if positional embeddings are reforged.
    # Guard: position_ids is 3D [batch, 3, seq_len] only during chunked prefill;
    # during decode it is 2D [batch, seq_len] and reforging is not needed.
    if (past_key_value is not None and getattr(past_key_value, "pos_embed_reforge", False)
            and position_ids is not None and position_ids.dim() == 3):
        # This code reforge the `position_ids` of current chunk,
        # the `position_ids` of previous chunks are reforged in KVCache.update()
        prev_tempo_idx = past_key_value.get_prev_temporal_idx(self.layer_idx)
        cur_tempo_idx = position_ids[0,0,0]
        if prev_tempo_idx + 1 != cur_tempo_idx:
            assert bsz == 1
            # print("Warning! Discontinuous positional ids %d (prev) + 1 != %d (cur) at layer %d. Fixed!" % (prev_tempo_idx,  cur_tempo_idx, self.layer_idx))
            # NOTE: clone `position_ids` to avoid influence of in-place ops in different layers
            position_ids = position_ids.clone()
            position_ids[0,0,:] += prev_tempo_idx + 1 - cur_tempo_idx
        position_embeddings = None # `position_embeddings` need to be re-calculated

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        cache_kwargs.update({"query_states": query_states, "position_ids": position_ids,
                             "rotary_emb": self.rotary_emb, "mrope_section": self.rope_scaling["mrope_section"]})
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # === Attention computation: dispatch based on transformers version ===
    from packaging import version
    import transformers as _tf
    _ge_457 = version.parse(_tf.__version__) >= version.parse("4.57.0")

    if _ge_457:
        # transformers >= 4.57: use the unified attention dispatch (flash/sdpa/eager)
        # This preserves the correct attention backend (flash_attention_2 etc.)
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import eager_attention_forward

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.head_dim**-0.5,
            sliding_window=getattr(self, 'sliding_window', None),
            position_ids=position_ids,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights
    else:
        # transformers < 4.57: use the original eager attention (for backward compat)
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def retake_Qwen2_5_VLForConditionalGeneration_segment_input_ids(self, input_ids):
    """Split video and text segments in the input_ids
    return: list[(s, e, type)], indices of [s, e) are of `type`.
    """
    videomask = (input_ids[0] == self.config.video_token_id)
    # Find the difference between consecutive elements
    diff = torch.diff(videomask.long())
    diff_pos_indices = (torch.where(diff == 1)[0] + 1).cpu().numpy()
    diff_neg_indices = (torch.where(diff == -1)[0] + 1).cpu().numpy()

    # True mask
    start_indices_true = diff_pos_indices
    end_indices_true = diff_neg_indices
    if videomask[0] == True: # segment starts at the beginning
        start_indices_true = np.insert(start_indices_true, 0, 0)
    if videomask[-1] == True: # segment ends at the beginning
        end_indices_true = np.append(end_indices_true, len(videomask))

    # False mask
    start_indices_flase = diff_neg_indices
    end_indices_flase = diff_pos_indices
    if videomask[0] == False:
        start_indices_flase = np.insert(start_indices_flase, 0, 0)
    if videomask[-1] == False:
        end_indices_flase = np.append(end_indices_flase, len(videomask))

    segments = (
        list(zip(start_indices_true, end_indices_true, ['video']*len(end_indices_true))) + 
        list(zip(start_indices_flase, end_indices_flase, ['text']*len(end_indices_flase)))
    )
    segments = sorted(segments, key=lambda x: x[0])
    return segments

def retake_Qwen2_5_VLForConditionalGeneration_get_chunk_size(self, config, video_grid_thw) -> int:
    # Calculate the number of tokens in each prefill chunk
    chunk_frames = (
        config.longvideo_kwargs.get('chunked_prefill_frames', None) if getattr(config, 'longvideo_kwargs', None) 
        else None
    )
    if chunk_frames is None:
        chunk_prefill_size = None
    else:
        T, H, W = video_grid_thw[0]
        t_factor = config.vision_config.spatial_merge_size**2 * config.vision_config.temporal_patch_size
        chunk_prefill_size = min(chunk_frames, T) * H * W // t_factor
        chunk_prefill_size = int(chunk_prefill_size.item())
        # Avoid machine error in ceil() when calculating `num_chunks`.
    return chunk_prefill_size

def bisection_projection(w, w_min, w_max, target_sum=None, tol=1e-5, max_iters=100):
    w = np.array(w, dtype=np.float64)
    if len(w) == 0:
        return []

    if not target_sum:
        target_sum = float(np.sum(w))

    if np.sum(w) <= 0:
        q = np.full_like(w, fill_value=target_sum / len(w), dtype=np.float64)
    else:
        q = w / np.sum(w) * target_sum

    mu_min = np.min(q) - w_max
    mu_max = np.max(q) - w_min

    for _ in range(max_iters):
        if mu_max - mu_min <= tol:
            break
        mu_mid = (mu_max + mu_min) / 2.0
        p_mid = np.clip(q - mu_mid, w_min, w_max)

        if np.sum(p_mid) > target_sum:
            mu_min = mu_mid
        else:
            mu_max = mu_mid

    mu_final = (mu_max + mu_min) / 2.0
    return np.clip(q - mu_final, w_min, w_max).tolist()


def compute_temporal_adaptation_ratios(config, inputs_embeds, modality_segments, video_grid_thw, chunk_size):
    """计算每个 chunk 的时序自适应压缩比（AdaReTaKe Eq. 9）。

    直接返回每个 chunk 的绝对 compression_ratio（已乘入 config 里的 compression_ratio），
    调用方（after_forward）可直接使用，无需再相乘。

    enable_temporal_adaptation=False 时，所有 chunk 均匀分配（返回等于 compression_ratio 的列表）。

    计算方式（enable_temporal_adaptation=True）：
    1. 将视频 embedding reshape 为 (num_frames_merged, patches_per_frame, dim)。
    2. 对空间维度取均值，得到每帧一个向量 (num_frames_merged, dim)。
    3. 对每个 chunk 取时间维度均值得到 chunk embedding，边界 padding 后计算与左右邻居的平均
       cosine distance：d_i = mean(1 - cos(chunk_i, chunk_{i-1}), 1 - cos(chunk_i, chunk_{i+1}))
       距离越大说明该 chunk 越独特，应分配更多 token（正向加权）。
    4. per_chunk_ratio_i = compression_ratio * d_i / mean(d)，使所有 chunk 的绝对压缩比均值
       等于 compression_ratio，整体预算不变。

    Returns:
        chunk_compression_ratios: List[float]，长度等于 num_chunks，每个元素是该 chunk 的绝对压缩比。
    """
    assert getattr(config, 'longvideo_kwargs', None) and config.longvideo_kwargs.get('kvcache_compression', False)
    compression_kwargs = config.longvideo_kwargs['kvcache_compression_kwargs']
    base_ratio = compression_kwargs['compression_ratio']
    enable_temporal_adaptation = compression_kwargs.get('enable_temporal_adaptation', False)

    # 先算 num_chunks（需要 video segment）
    for seg_id, (s, e, dtype) in enumerate(modality_segments):
        if dtype != 'video':
            continue

        num_video_tokens = e - s
        num_chunks = math.ceil(num_video_tokens / chunk_size)

        if not enable_temporal_adaptation:
            # 均匀分配，每个 chunk 直接用 base_ratio
            return [base_ratio] * num_chunks

        video_segment_embeds = inputs_embeds[0, s:e]  # [num_video_tokens, hidden_dim]

        # Reshape 为 (num_frames_merged, patches_per_frame, hidden_dim)
        grid_t, grid_h, grid_w = video_grid_thw[0]
        t_factor = config.vision_config.spatial_merge_size ** 2 * config.vision_config.temporal_patch_size
        num_frames_merged = int((grid_t * grid_h * grid_w / t_factor) // (grid_h * grid_w / t_factor))
        patches_per_frame = int(grid_h * grid_w / t_factor)
        frame_embeds = video_segment_embeds[:num_frames_merged * patches_per_frame].reshape(
            num_frames_merged, patches_per_frame, -1
        )  # [num_frames_merged, patches_per_frame, dim]

        # 对空间维度取均值，得到每帧一个向量
        frame_embeds = frame_embeds.mean(dim=1)  # [num_frames_merged, dim]

        # 按 chunk 划分帧，对时间维度取均值得到每个 chunk 一个向量
        frames_per_chunk = num_frames_merged / num_chunks  # float，均匀分配
        chunk_embeds = []
        for idx in range(num_chunks):
            fs = int(idx * frames_per_chunk)
            fe = min(int((idx + 1) * frames_per_chunk), num_frames_merged)
            chunk_embeds.append(frame_embeds[fs:fe].mean(dim=0))  # [dim]
        chunk_embeds = torch.stack(chunk_embeds)  # [num_chunks, dim]

        # 边界 padding：复制首尾 chunk embedding，使每个 chunk 都有左右邻居
        padded = torch.cat([chunk_embeds[:1], chunk_embeds, chunk_embeds[-1:]], dim=0)  # [num_chunks+2, dim]

        # d_i = mean(1 - cos(chunk_i, chunk_{i-1}), 1 - cos(chunk_i, chunk_{i+1}))
        # 距离越大说明该 chunk 越独特，应分配更多 token（正向加权）
        center = padded[1:-1]   # [num_chunks, dim]
        left   = padded[:-2]    # [num_chunks, dim]
        right  = padded[2:]     # [num_chunks, dim]
        dist_left  = 1 - F.cosine_similarity(center, left,  dim=-1)  # [num_chunks]
        dist_right = 1 - F.cosine_similarity(center, right, dim=-1)  # [num_chunks]
        chunk_distances = ((dist_left + dist_right) / 2).tolist()

        # per_chunk_ratio_i = base_ratio * d_i / mean(d)，使均值 = base_ratio
        mean_distance = sum(chunk_distances) / num_chunks
        if mean_distance > 0:
            # chunk_compression_ratios = [base_ratio * d / mean_distance for d in chunk_distances]
            raw_chunk_compression_ratios = [base_ratio * d / mean_distance for d in chunk_distances]
            chunk_compression_ratios = bisection_projection(
                raw_chunk_compression_ratios,
                w_min=1e-3,
                w_max=1.0,
                target_sum=base_ratio * num_chunks,
            )
        else:
            chunk_compression_ratios = [base_ratio] * num_chunks

        # # ── 诊断日志：收集 temporal adaptation 分布数据 ──
        # # 输出到 /tmp/temporal_ratios_log.jsonl，每行一个样本
        # _log_path = os.environ.get('TEMPORAL_RATIO_LOG', '')
        # if _log_path:
        #     import json as _json
        #     try:
        #         _entry = {
        #             'base_ratio': float(base_ratio),
        #             'num_chunks': num_chunks,
        #             'mean_distance': float(mean_distance),
        #             'chunk_distances': chunk_distances,
        #             'chunk_ratios': chunk_compression_ratios,
        #             'min_ratio': min(chunk_compression_ratios),
        #             'max_ratio': max(chunk_compression_ratios),
        #         }
        #         with open(_log_path, 'a') as _f:
        #             _f.write(_json.dumps(_entry) + '\n')
        #     except Exception:
        #         pass  # 静默失败，不影响推理

        return chunk_compression_ratios

    return None  # 没有 video segment（不应发生）

def retake_Qwen2_5_VLForConditionalGeneration_forge_input_chunks(self, ss, ee, modality_segments, cache_position, position_ids, attention_mask, past_key_values, inputs_embeds):
    cache_position_chunk = cache_position[ss:ee]
    position_ids_chunk = position_ids[:,:,ss:ee]
    attention_mask_chunk = attention_mask[:,:ee] # NOTE: specially from 0 to ee
    inputs_embeds_chunk = inputs_embeds[:,ss:ee]
    prompt_length = None

    if getattr(self.config, 'longvideo_kwargs', None) and self.config.longvideo_kwargs.get('kvcache_compression', False):
        compression_kwargs = self.config.longvideo_kwargs['kvcache_compression_kwargs']
        if compression_kwargs.get('prompt_guided_compression', False) and compression_kwargs.get('compression_ratio', 1) < 1.0:
            # Prompt guided KV cache compression
            s_p, e_p, t_p = modality_segments[-1]
            max_guide_length = min(e_p - s_p, compression_kwargs.get('max_guide_length', 999999999999999999))
            s_p = s_p + (e_p - s_p - max_guide_length)
            assert t_p == 'text'
            pos_offset = position_ids[0,0,s_p] - position_ids_chunk[0,0,-1] - 1 # (3, bs, seq_len)
            position_ids_chunk = torch.cat([position_ids_chunk, position_ids[:,:,s_p:e_p] - pos_offset], dim=2)
            attention_mask_chunk = torch.cat([attention_mask_chunk, attention_mask[:,s_p:e_p]], dim=1)
            inputs_embeds_chunk = torch.cat([inputs_embeds_chunk, inputs_embeds[:,s_p:e_p]], dim=1)
            prompt_length = e_p - s_p
            cache_position_chunk = cache_position[ss:ee+prompt_length]

    return cache_position_chunk, position_ids_chunk, attention_mask_chunk, inputs_embeds_chunk, prompt_length


def retake_Qwen2_5_VLForConditionalGeneration_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    assert input_ids.shape[0] == 1, "Batch inference of long video is not supported yet!"

    # ── Compatibility shim: transformers >= 4.57 restructured the model ──────
    # Old: self.visual, self.model (LLM), self.model.embed_tokens, self.config.vision_config
    # New: self.model.visual, self.model.language_model (LLM),
    #      self.model.language_model.embed_tokens, self.config.vision_config (unchanged)
    _new_arch = hasattr(self.model, 'language_model')
    _visual         = self.model.visual if _new_arch else self.visual
    _llm            = self.model.language_model if _new_arch else self.model
    _embed_tokens   = _llm.embed_tokens
    _vision_config  = self.config.vision_config
    _vocab_size     = (self.config.text_config.vocab_size if _new_arch
                       else self.config.vocab_size)
    # rope_deltas stored on self.model in new arch, on self in old arch
    def _get_rope_deltas():
        return self.model.rope_deltas if _new_arch else self.rope_deltas
    def _set_rope_deltas(v):
        if _new_arch: self.model.rope_deltas = v
        else:         self.rope_deltas = v
    # ─────────────────────────────────────────────────────────────────────────

    if (cache_position is not None and cache_position[0] == 0): # Prefill
        is_prefill = True
        # Calculate chunk size based on inputs
        chunk_size = self.get_chunk_size(self.config, video_grid_thw)
        # Configuring KV Cache compression kwargs
        if getattr(self.config, 'longvideo_kwargs', None) and self.config.longvideo_kwargs.get('kvcache_compression', False):
            compression_kwargs = self.config.longvideo_kwargs['kvcache_compression_kwargs']
            if compression_kwargs.get('dynamic_compression_ratio', False):
                # Dynamic compression ratio
                input_length = input_ids.shape[1]
                max_input_length = compression_kwargs['max_input_length']
                if input_length <= max_input_length:
                    compression_kwargs['compression_ratio'] = 1
                else:
                    compression_kwargs['compression_ratio'] = max_input_length / input_length
        if chunk_size is not None:
            modality_segments = self.segment_input_ids(input_ids)
            past_key_values = build_kvcache(self.config)
            use_cache = True
    else:
        is_prefill = False
        chunk_size = None

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (cache_position is not None and cache_position[0] == 0) or _get_rope_deltas() is None:
            # get_rope_index moved from ForConditionalGeneration to Qwen2_5_VLModel in transformers >= 4.57
            _get_rope_index = self.model.get_rope_index if _new_arch else self.get_rope_index
            position_ids, rope_deltas = _get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            _set_rope_deltas(rope_deltas)
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length = input_ids.shape
            delta = (
                (cache_position[0] + _get_rope_deltas()).to(input_ids.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    if inputs_embeds is None:
        # Helper: transformers 4.57 的 visual encoder 返回 BaseModelOutputWithPooling 而非纯 tensor
        def _extract_embeds(output):
            return output.last_hidden_state if hasattr(output, 'last_hidden_state') else output

        # Extract visual features
        if pixel_values is not None:
            pixel_values = pixel_values.type(_visual.dtype)
            image_embeds = _extract_embeds(_visual(pixel_values, grid_thw=image_grid_thw))

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(_visual.dtype)
            grid_t, grid_h, grid_w = video_grid_thw[0]
            # NOTE: Split video into chunks to avoid OOM due to large activations during visual forward
            # chunk_size can be up to 128 or higher if you have flash attention
            frame_chunk_size = getattr(self.config, 'longvideo_kwargs', {}).get('frame_chunk_size', 1000000000)
            if grid_t < frame_chunk_size:
                video_embeds = _extract_embeds(_visual(pixel_values_videos, grid_thw=video_grid_thw))
            else:
                d = pixel_values_videos.shape[-1]
                pixel_values_videos = pixel_values_videos.reshape(grid_t, grid_h*grid_w, d)
                video_embeds = []
                for i in range(0, grid_t, frame_chunk_size):
                    pixel_values_videos_chunk = pixel_values_videos[i:i+frame_chunk_size]
                    grid_t_chunk = pixel_values_videos_chunk.shape[0]
                    video_grid_thw_chunk = video_grid_thw.clone()
                    video_grid_thw_chunk[0,0] = grid_t_chunk
                    video_embeds.append(
                        _extract_embeds(_visual(pixel_values_videos_chunk.reshape(-1, d), grid_thw=video_grid_thw_chunk))
                    )
                video_embeds = torch.cat(video_embeds)

        # Concat visual and textual features
        inputs_embeds = _embed_tokens(input_ids)
        if pixel_values is not None:
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
        if position_ids is not None:
            position_ids = position_ids.to(inputs_embeds.device)

    if is_prefill and chunk_size is not None: # Chunked prefill stage
        assert past_key_values is not None
        kvcache_compression = getattr(past_key_values, 'kvcache_compression', False)

        # Pre-compute per-chunk temporal adaptation ratios (AdaReTaKe Eq. 9)
        if kvcache_compression:
            chunk_compression_ratios = compute_temporal_adaptation_ratios(
                self.config, inputs_embeds, modality_segments, video_grid_thw, chunk_size
            )

        for seg_id, (s, e, dtype) in enumerate(modality_segments):
            if dtype == 'text': # Prefill text without kvcache_compression
                past_key_values.kvcache_compression = False
                outputs = _llm(
                    input_ids=None,
                    position_ids=position_ids[:,:,s:e],
                    attention_mask=attention_mask[:,:e],
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds[:,s:e],
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position[s:e],
                )
                past_key_values = outputs['past_key_values']
            elif dtype == 'video': # Prefill video may with kvcache_compression
                num_chunks = math.ceil((e - s) / chunk_size)
                past_key_values.kvcache_compression = kvcache_compression
                for idx in tqdm(range(num_chunks), total=num_chunks, desc='Prefilling chunk', disable=not DEBUG_MODE):
                    ss = s + idx * chunk_size
                    ee = min(s + (idx + 1) * chunk_size, e)
                    # if keypatches_mask is not None:
                    #     past_key_values.keypatches_mask_chunk = keypatches_mask[ss:ee]
                    cache_position_chunk, position_ids_chunk, attention_mask_chunk, inputs_embeds_chunk, prompt_length = self.forge_input_chunks(
                        ss, ee, modality_segments, cache_position, position_ids, attention_mask, past_key_values, inputs_embeds
                    )
                    if hasattr(past_key_values, 'before_forward'):
                        past_key_values.before_forward(prompt_length=prompt_length, position_ids=position_ids_chunk)
                    # Pass pre-computed temporal adaptation ratio for this chunk
                    if kvcache_compression and hasattr(past_key_values, 'set_temporal_adaptation_ratio'):
                        past_key_values.set_temporal_adaptation_ratio(chunk_compression_ratios[idx])
                    outputs = _llm(
                        input_ids=None,
                        position_ids=position_ids_chunk,
                        attention_mask=attention_mask_chunk,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds_chunk,
                        use_cache=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        cache_position=cache_position_chunk,
                    )
                    past_key_values = outputs['past_key_values']
                    if hasattr(past_key_values, 'after_forward'):
                        past_key_values.after_forward()
                past_key_values.keypatches_mask = None
                past_key_values.kvcache_compression = False # Turned off for decoding
            else:
                raise ValueError
    else: # Decode / Standard prefill stage
        outputs = _llm(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, _vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=_get_rope_deltas(),
    )