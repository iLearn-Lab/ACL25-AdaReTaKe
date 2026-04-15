import transformers

from retake.qwen2_vl import (
    retake_Qwen2VLAttention_forward,
    retake_Qwen2VLForConditionalGeneration_compress_video_tokens,
    retake_Qwen2VLForConditionalGeneration_segment_input_ids,
    retake_Qwen2VLForConditionalGeneration_get_chunk_size,
    retake_Qwen2VLForConditionalGeneration_forge_input_chunks,
    retake_Qwen2VLForConditionalGeneration_forward,
)
from retake.qwen2_5_vl import (
    retake_Qwen2_5_VLAttention_forward,
    retake_Qwen2_5_VLForConditionalGeneration_segment_input_ids,
    retake_Qwen2_5_VLForConditionalGeneration_get_chunk_size,
    retake_Qwen2_5_VLForConditionalGeneration_forge_input_chunks,
    retake_Qwen2_5_VLForConditionalGeneration_forward,
)
from retake.llava_onevision import (
    retake_Qwen2Attention_init,
    retake_Qwen2Attention_forward,
    retake_LlavaOnevisionForConditionalGeneration_get_chunk_size,
    retake_LlavaOnevisionForConditionalGeneration_segment_input_ids,
    retake_LlavaOnevisionForConditionalGeneration_compress_video_tokens,
    retake_LlavaOnevisionForConditionalGeneration_forge_input_chunks,
    retake_LlavaOnevisionForConditionalGeneration_forward,
)


def patch_qwen2vl_config(config, exp_configs):
    # Rope Scaling
    if 'scaling_factor' in exp_configs:
        # transformers 4.57: rope_scaling 是 property，不能原地修改，需要整体赋值
        # 保留原始 rope_scaling 中的 mrope_section 等字段，仅覆盖 rope_type/factor 等
        old_rope = dict(config.rope_scaling) if config.rope_scaling else {}
        old_rope.pop('type', None)  # 删除旧版 type 字段
        old_rope.update({
            'rope_type': 'yarn',
            'factor': exp_configs['scaling_factor'],
            'beta_fast': 32.0,
            'beta_slow': 1.0,
        })
        config.rope_scaling = old_rope
    # ReTaKe
    config.longvideo_kwargs = exp_configs.get('longvideo_kwargs', {})
    return config


def patch_qwen2_5_vl_config(config, exp_configs):
    # Rope Scaling
    if 'scaling_factor' in exp_configs:
        # transformers 4.57: rope_scaling 是 property，不能原地修改，需要整体赋值
        # 保留原始 rope_scaling 中的 mrope_section 等字段，仅覆盖 rope_type/factor 等
        old_rope = dict(config.rope_scaling) if config.rope_scaling else {}
        old_rope.pop('type', None)  # 删除旧版 type 字段
        old_rope.update({
            'rope_type': 'yarn',
            'factor': exp_configs['scaling_factor'],
            'beta_fast': 32.0,
            'beta_slow': 1.0,
        })
        config.rope_scaling = old_rope
    # ReTaKe
    config.longvideo_kwargs = exp_configs.get('longvideo_kwargs', {})
    return config


def patch_llava_onevision_config(config, exp_configs):
    # Rope Scaling
    if 'scaling_factor' in exp_configs:
        config.text_config.rope_scaling = {
            'rope_type': 'yarn',
            'factor': exp_configs['scaling_factor'],
            'beta_fast': 32.0,
            'beta_slow': 1.0,
        }
    # ReTaKe
    config.longvideo_kwargs = exp_configs.get('longvideo_kwargs', {})
    return config


def patch_qwen2vl(method):

    if method == "retake":
        print("Using ReTaKe for Qwen2VLForConditionalGeneration!")
        # transformers >= 4.57: 只有统一的 Qwen2VLAttention，Sdpa/FlashAttention2 子类已移除
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention.forward = retake_Qwen2VLAttention_forward
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.compress_video_tokens = retake_Qwen2VLForConditionalGeneration_compress_video_tokens
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.segment_input_ids = retake_Qwen2VLForConditionalGeneration_segment_input_ids
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.get_chunk_size = retake_Qwen2VLForConditionalGeneration_get_chunk_size
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forge_input_chunks = retake_Qwen2VLForConditionalGeneration_forge_input_chunks
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = retake_Qwen2VLForConditionalGeneration_forward
    else:
        raise NotImplementedError


def patch_qwen2_5_vl(method):

    if method == "retake":
        print("Using ReTaKe for Qwen2_5_VLForConditionalGeneration!")
        # transformers >= 4.57: 只有统一的 Qwen2_5_VLAttention，Sdpa/FlashAttention2 子类和
        # _prepare_4d_causal_attention_mask_with_cache_position 方法均已移除
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = retake_Qwen2_5_VLAttention_forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.segment_input_ids = retake_Qwen2_5_VLForConditionalGeneration_segment_input_ids
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_chunk_size = retake_Qwen2_5_VLForConditionalGeneration_get_chunk_size
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forge_input_chunks = retake_Qwen2_5_VLForConditionalGeneration_forge_input_chunks
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = retake_Qwen2_5_VLForConditionalGeneration_forward
    else:
        raise NotImplementedError


def patch_llava_onevision(method):

    if method == "retake":
        print("Using ReTaKe for LlavaOnevisionForConditionalGeneration!")
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.__init__ = retake_Qwen2Attention_init
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = retake_Qwen2Attention_forward
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.get_chunk_size = retake_LlavaOnevisionForConditionalGeneration_get_chunk_size
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.segment_input_ids = retake_LlavaOnevisionForConditionalGeneration_segment_input_ids
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.compress_video_tokens = retake_LlavaOnevisionForConditionalGeneration_compress_video_tokens
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.forge_input_chunks = retake_LlavaOnevisionForConditionalGeneration_forge_input_chunks
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.forward = retake_LlavaOnevisionForConditionalGeneration_forward
    else:
        raise NotImplementedError
