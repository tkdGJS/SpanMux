# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""

import time
from collections.abc import Generator, Iterable
from itertools import islice

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.isolation_memory import isolation_execution_enabled, isolation_memory_enabled
from vllm.platforms import current_platform
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention.encoder_only_attention import (
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.llama3_rope import Llama3RotaryEmbedding
from vllm.model_executor.layers.green_context import (
    pd_layerwise_handoff_layer_boundary,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.operator_selective_attention import (
    get_pd_phase_plan,
    maybe_operator_selective_pd_attention,
    muxwise_phase_baseline_enabled,
    muxwise_phase_green_context_enabled,
    operator_selective_pd_enabled,
    operator_selective_pd_overlap_enabled,
    record_operator_selective_pd_fallback,
    run_pd_pair,
    run_with_pd_phase_context,
)
from vllm.model_executor.layers.operator_memory_shaping import (
    opmux_enabled,
    record_opmux_attention,
    record_opmux_mlp,
    record_opmux_o_proj,
    record_opmux_qkv,
)
from vllm.model_executor.layers.pd_layerwise_handoff import (
    layerwise_profile_handoff_requested,
)
from vllm.model_executor.layers.pd_prefill_workspace import (
    active_prefill_workspace_buffer,
    prepare_active_llama_prefill_workspace,
    prefill_workspace_attention,
    prefill_workspace_ring_sharing_requested,
    prefill_workspace_requested,
)
from vllm.model_executor.layers.kv_retention_resume_frame import (
    KVRetentionLayerContinuation,
    KVRetentionModelState,
    kv_retention_resume_frame_requested,
)
from vllm.model_executor.layers.kv_retention_parked_prefill import (
    kv_retention_p1_parked_requested, kv_retention_bound_prefill_submission,
)
from vllm.model_executor.layers.ttft_layerwise_prefill import (
    TTFTLayerwisePrefillBoundary,
    TTFTLayerwisePrefillExecution,
    TTFTLayerwisePrefillModelYield,
    active_ttft_layerwise_prefill_execution,
    ttft_layerwise_prefill_execution_requested,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType


def _isolation_fused_norm_requested() -> bool:
    # Do not override Inductor's normal fusion on compiled Native models.
    return (
        isolation_memory_enabled("FUSED_RMS_NORM")
        and current_platform.is_cuda()
        and (prefill_workspace_requested()
             or layerwise_profile_handoff_requested()
             or ttft_layerwise_prefill_execution_requested())
    )

from .adapters import as_embedding_model, as_seq_cls_model
from .interfaces import (
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
        disable_tp: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            disable_tp=disable_tp,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=disable_tp,
            prefix=f"{prefix}.down_proj",
        )
        self.opmux_layer_name = prefix
        self.opmux_enabled = opmux_enabled()
        self.isolation_extended_workspace = isolation_memory_enabled(
            "PREFILL_EXTENDED_WORKSPACE"
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x, *, pd_prefill_workspace: bool = False):
        if not self.opmux_enabled:
            if not pd_prefill_workspace:
                x, _ = self.gate_up_proj(x)
                x = self.act_fn(x)
                x, _ = self.down_proj(x)
                return x

            gate_up_workspace = active_prefill_workspace_buffer(
                "gate_up",
                x,
                width=self.gate_up_proj.weight.shape[0],
            )
            if gate_up_workspace is None:
                x, _ = self.gate_up_proj(x)
            else:
                x, _ = self.gate_up_proj.forward_into(x, gate_up_workspace)
            activation_workspace = active_prefill_workspace_buffer(
                "activation",
                x,
                width=x.shape[-1] // 2,
            )
            if activation_workspace is None:
                x = self.act_fn(x)
            else:
                x = self.act_fn.forward_into(x, activation_workspace)
            down_workspace = None
            if self.isolation_extended_workspace:
                down_workspace = active_prefill_workspace_buffer(
                    "hidden", x, width=self.down_proj.weight.shape[0]
                )
            if down_workspace is None:
                x, _ = self.down_proj(x)
            else:
                x, _ = self.down_proj.forward_into(x, down_workspace)
            return x

        input_tensor = x
        gate_up, _ = self.gate_up_proj(x)
        activation = self.act_fn(gate_up)
        output, _ = self.down_proj(activation)
        record_opmux_mlp(
            self.opmux_layer_name,
            input_tensor,
            gate_up,
            activation,
            output,
        )
        return output


class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_size // self.total_num_heads
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        llama_4_scaling_config = getattr(config, "llama_4_scaling", None)
        self.do_llama_4_scaling = llama_4_scaling_config is not None
        if self.do_llama_4_scaling:
            self.llama_4_scaling_original_max_position_embeddings = (
                llama_4_scaling_config["original_max_position_embeddings"]
            )
            self.llama_4_scaling_beta = llama_4_scaling_config["beta"]

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self._init_rotary_emb(config, quant_config=quant_config)

        sliding_window = None
        if layer_types := getattr(config, "layer_types", None):
            # Fix for Eagle3 compatibility:
            # for draft models, subtract target layer count
            # to get draft-relative layer index starting from 0
            if hasattr(config, "target_layer_count"):
                # This is a draft model,
                # adjust layer_idx to be relative to draft layers
                effective_layer_idx = layer_idx - config.target_layer_count
            else:
                # This is a target model, use layer_idx directly
                effective_layer_idx = layer_idx
            assert effective_layer_idx < len(layer_types), (
                f"effective_layer_idx: {effective_layer_idx} "
                f"is out of bounds for layer_types: {layer_types}"
            )

            is_sliding = layer_types[effective_layer_idx] == "sliding_attention"
            if is_sliding:
                sliding_window = config.sliding_window

        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )

        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )
        self.opmux_layer_name = prefix
        self.opmux_enabled = opmux_enabled()
        self.operator_selective_pd_enabled = (
            attn_cls is Attention and operator_selective_pd_enabled()
        )
        self.isolation_extended_workspace = isolation_memory_enabled(
            "PREFILL_EXTENDED_WORKSPACE"
        )
        self.isolation_prefill_fused_rope = (
            isolation_execution_enabled("PREFILL_FUSED_ROPE")
            and current_platform.is_cuda()
            and type(self.rotary_emb) in (RotaryEmbedding, Llama3RotaryEmbedding)
        )
        self.isolation_ring_workspace_sharing = prefill_workspace_ring_sharing_requested()

    def _apply_rotary_emb(self, positions, q, k, *, pd_prefill_workspace):
        if (pd_prefill_workspace and self.isolation_prefill_fused_rope
                and not self.opmux_enabled and q.is_cuda):
            # Q/K are private to this live Prefill workspace generation. Use
            # the existing in-place op; never mutate the globally cached RoPE
            # object's dispatch method (Decode graphs may share that object).
            return self.rotary_emb.forward_cuda(positions, q, k)
        return self.rotary_emb(positions, q, k)

    def _get_llama_4_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        # Llama4 scaling
        scaling = 1 + self.llama_4_scaling_beta * torch.log(
            1
            + torch.floor(
                positions / self.llama_4_scaling_original_max_position_embeddings
            )
        )
        # Broadcast over head_dim
        return scaling.unsqueeze(-1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        pd_prefill_workspace: bool = False,
    ) -> torch.Tensor:
        if pd_prefill_workspace and not self.opmux_enabled:
            qkv_width = self.q_size + 2 * self.kv_size
            qkv_workspace = active_prefill_workspace_buffer(
                "qkv",
                hidden_states,
                width=qkv_width,
            )
            if qkv_workspace is None:
                qkv, _ = self.qkv_proj(hidden_states)
            else:
                qkv, _ = self.qkv_proj.forward_into(
                    hidden_states,
                    qkv_workspace,
                )
        else:
            qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.opmux_enabled:
            record_opmux_qkv(self.opmux_layer_name, hidden_states, qkv, q, k, v)
        if pd_prefill_workspace and self.isolation_prefill_fused_rope:
            q, k = self._apply_rotary_emb(
                positions, q, k, pd_prefill_workspace=True)
        else:
            q, k = self.rotary_emb(positions, q, k)
        if self.do_llama_4_scaling:
            attn_scale = self._get_llama_4_attn_scale(positions)
            q = (q * attn_scale).to(q.dtype)
        attn_output = None
        if self.operator_selective_pd_enabled:
            attn_output = maybe_operator_selective_pd_attention(
                self.attn, q, k, v
            )
        if attn_output is None:
            attention_workspace = None
            if pd_prefill_workspace and not self.opmux_enabled and bool(
                getattr(self.attn, "use_output", False)
            ):
                attention_workspace = active_prefill_workspace_buffer(
                    "attention",
                    q,
                    width=self.q_size,
                )
            if pd_prefill_workspace and self.isolation_ring_workspace_sharing:
                # QKV is already produced; norm/MLP scratch is dead until
                # Attention returns. Never extend this scope over o_proj.
                with prefill_workspace_attention():
                    attn_output = self.attn(q, k, v, output=attention_workspace)
            else:
                attn_output = self.attn(q, k, v, output=attention_workspace)
        if self.opmux_enabled:
            record_opmux_attention(self.opmux_layer_name, q, k, v, attn_output)
        projection_workspace = None
        if (pd_prefill_workspace and not self.opmux_enabled
                and self.isolation_extended_workspace):
            projection_workspace = active_prefill_workspace_buffer(
                "norm", attn_output, width=self.o_proj.weight.shape[0]
            )
        if projection_workspace is None:
            output, _ = self.o_proj(attn_output)
        else:
            output, _ = self.o_proj.forward_into(attn_output, projection_workspace)
        if self.opmux_enabled:
            record_opmux_o_proj(self.opmux_layer_name, attn_output, output)
        return output

    def _init_rotary_emb(
        self,
        config: LlamaConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
            is_neox_style=is_neox_style,
        )


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
    ) -> None:
        super().__init__()

        config = config or vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = self.get_quant_config(vllm_config)

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        bias_o_proj = attention_bias
        # support internlm/internlm3-8b with qkv_bias
        if hasattr(config, "qkv_bias"):
            attention_bias = config.qkv_bias

        # By default, Llama uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. parasail-ai/GritLM-7B-vllm)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.isolation_extended_workspace = isolation_memory_enabled(
            "PREFILL_EXTENDED_WORKSPACE"
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
            enforce_enable=_isolation_fused_norm_requested(),
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
            enforce_enable=_isolation_fused_norm_requested(),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase_result = self._maybe_muxwise_phase_forward(
            positions, hidden_states, residual
        )
        if phase_result is not None:
            return phase_result

        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def forward_with_pd_prefill_workspace(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense Llama layer path using the Future-owned scratch arena."""

        norm_workspace = None
        if self.isolation_extended_workspace:
            norm_workspace = active_prefill_workspace_buffer(
                "norm", hidden_states, width=self.hidden_size
            )
        if residual is None:
            residual = hidden_states
            hidden_states = (
                self.input_layernorm(hidden_states) if norm_workspace is None else
                self.input_layernorm.forward_into(hidden_states, norm_workspace)
            )
        elif norm_workspace is not None:
            hidden_states, residual = self.input_layernorm.forward_into(
                hidden_states, norm_workspace, residual
            )
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states,
                residual,
            )
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            pd_prefill_workspace=True,
        )
        if norm_workspace is None:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        else:
            hidden_states, residual = self.post_attention_layernorm.forward_into(
                hidden_states, norm_workspace, residual
            )
        hidden_states = self.mlp(
            hidden_states,
            pd_prefill_workspace=True,
        )
        return hidden_states, residual

    def _run_phase_layer(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            phase_residual = hidden_states
            phase_hidden_states = self.input_layernorm(hidden_states)
        else:
            phase_hidden_states, phase_residual = self.input_layernorm(
                hidden_states, residual
            )
        phase_hidden_states = self.self_attn(
            positions=positions, hidden_states=phase_hidden_states
        )
        phase_hidden_states, phase_residual = self.post_attention_layernorm(
            phase_hidden_states, phase_residual
        )
        phase_hidden_states = self.mlp(phase_hidden_states)
        return phase_hidden_states, phase_residual

    def _maybe_muxwise_phase_forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not muxwise_phase_baseline_enabled():
            return None

        layer_name = getattr(self.self_attn.attn, "layer_name", "")
        plan = get_pd_phase_plan(layer_name)
        if plan is None:
            return None
        if plan.num_actual_tokens != hidden_states.shape[0]:
            record_operator_selective_pd_fallback(
                "phase_baseline_padded_tokens_unsupported"
            )
            return None

        decode_slice = slice(0, plan.num_decode_tokens)
        prefill_slice = slice(plan.num_decode_tokens, plan.num_actual_tokens)

        def decode_fn() -> tuple[torch.Tensor, torch.Tensor]:
            return run_with_pd_phase_context(
                layer_name,
                plan.decode_metadata,
                lambda: self._run_phase_layer(
                    positions[decode_slice],
                    hidden_states[decode_slice],
                    None if residual is None else residual[decode_slice],
                ),
            )

        def prefill_fn() -> tuple[torch.Tensor, torch.Tensor]:
            return run_with_pd_phase_context(
                layer_name,
                plan.prefill_metadata,
                lambda: self._run_phase_layer(
                    positions[prefill_slice],
                    hidden_states[prefill_slice],
                    None if residual is None else residual[prefill_slice],
                ),
            )

        decode_result, prefill_result = run_pd_pair(
            decode_fn,
            prefill_fn,
            kind="phase",
            plan=plan,
            use_green_context=muxwise_phase_green_context_enabled(),
            overlap=operator_selective_pd_overlap_enabled(),
        )
        decode_hidden_states, decode_residual = decode_result
        prefill_hidden_states, prefill_residual = prefill_result
        return (
            torch.cat((decode_hidden_states, prefill_hidden_states), dim=0),
            torch.cat((decode_residual, prefill_residual), dim=0),
        )

    def get_quant_config(self, vllm_config: VllmConfig) -> QuantizationConfig | None:
        """Get quantization config for this layer. Override in subclasses."""
        return vllm_config.quant_config


def llama_model_invariants(
    input_ids, positions, intermediate_tensors=None, inputs_embeds=None
):
    """Shape invariants for Llama model compilation, those are translated to
    runtime assertions for unbacked dynamic shapes and are compiled away for
    backed"""
    if input_ids is not None:
        torch._check(positions.size()[0] == input_ids.size()[0])


def llama_model_torch_compile_enabled(vllm_config: VllmConfig) -> bool:
    """Keep real eager boundaries for layerwise execution or fixed scratch.

    The layerwise handoff callback reads process-local control state at every
    decoder-layer boundary.  A full-graph TorchDynamo trace cannot represent
    that ``ContextVar`` lookup, and compiling the callback away would also
    remove the runtime scheduling boundary.  The outer model runner may still
    capture the eager model as a Decode-only CUDA graph; phase-pure Prefill
    remains eager so its handoff callbacks stay host-visible.  The fixed
    Prefill workspace has the same requirement because its caller-owned output
    views and Future lease are runtime state rather than graph constants.
    """

    del vllm_config
    layerwise_requested = layerwise_profile_handoff_requested()
    ttft_layer_groups_requested = (
        ttft_layerwise_prefill_execution_requested()
    )
    workspace_requested = prefill_workspace_requested()
    enabled = (
        not layerwise_requested
        and not ttft_layer_groups_requested
        and not workspace_requested
    )
    if not enabled:
        logger.info_once(
            "Layerwise Prefill execution/handoff or fixed Prefill workspace "
            "keeps Llama eager; Decode-only CUDA graph capture remains "
            "available through the outer model wrapper."
        )
    return enabled


@support_torch_compile(
    # TODO[#32068]: Investigate recompilation
    # mark_unbacked_dims={"input_ids": 0},
    shape_invariants=llama_model_invariants,
    enable_if=llama_model_torch_compile_enabled,
)
class LlamaModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps,
                enforce_enable=_isolation_fused_norm_requested(),
            )
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers = tuple[int, ...]()
        # Default-off.  The enabled path requires eager execution so this
        # Python layer boundary remains a real host-visible scheduling point.
        self.pd_layerwise_profile_handoff_enabled = (
            layerwise_profile_handoff_requested()
        )
        # Use the same static predicate as compile eligibility. Even reading
        # the optional execution ContextVar breaks fullgraph tracing when the
        # feature is disabled (e.g. the native mixed-batch SRPT control).
        self.ttft_layerwise_prefill_execution_enabled = (
            ttft_layerwise_prefill_execution_requested()
        )
        self.kv_retention_p1_parked_enabled = kv_retention_p1_parked_requested()
        self.isolation_p1_boundary_fastpath = isolation_execution_enabled(
            "P1_BOUNDARY_FASTPATH")
        self.pd_prefill_workspace_enabled = prefill_workspace_requested()
        tp_size = get_tensor_model_parallel_world_size()
        total_num_heads = int(config.num_attention_heads)
        total_num_kv_heads = int(
            getattr(config, "num_key_value_heads", total_num_heads)
        )
        head_dim = int(
            getattr(config, "head_dim", None)
            or config.hidden_size // total_num_heads
        )
        self.pd_prefill_workspace_tp_size = tp_size
        self.pd_prefill_workspace_qkv_width = (
            total_num_heads // tp_size
            + 2 * max(1, total_num_kv_heads // tp_size)
        ) * head_dim
        self.pd_prefill_workspace_intermediate_size = int(
            config.intermediate_size
        )
        self.pd_prefill_workspace_lora_enabled = vllm_config.lora_config is not None
        self.pd_prefill_workspace_opmux_enabled = opmux_enabled()
        self.pd_prefill_workspace_mixed_phase_enabled = (
            muxwise_phase_baseline_enabled()
        )
        self.pd_prefill_workspace_layer_path_supported = all(
            hasattr(self.layers[layer_index], "forward_with_pd_prefill_workspace")
            for layer_index in range(self.start_layer, self.end_layer)
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _forward_decoder_layer_range(
        self,
        *,
        start_layer: int,
        end_layer: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        aux_hidden_states: list[torch.Tensor],
        workspace_active: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Execute one checked half-open decoder-layer range.

        This helper is used only by the default-off TTFT layer-group path.
        The legacy path below intentionally retains its original loop so
        disabling the flag restores the exact prior execution structure.
        """

        if not (
            self.start_layer <= start_layer < end_layer <= self.end_layer
        ):
            raise RuntimeError(
                "TTFT Prefill layer range is outside this pipeline rank: "
                f"[{start_layer}, {end_layer}) not within "
                f"[{self.start_layer}, {self.end_layer})"
            )
        for layer_index in range(start_layer, end_layer):
            relative_index = layer_index - self.start_layer
            if relative_index in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
            layer = self.layers[layer_index]
            if workspace_active:
                hidden_states, residual = (
                    layer.forward_with_pd_prefill_workspace(
                        positions,
                        hidden_states,
                        residual,
                    )
                )
            else:
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    residual,
                )
            if self.pd_layerwise_profile_handoff_enabled:
                pd_layerwise_handoff_layer_boundary(layer_index)
        return hidden_states, residual

    def _kv_retention_advance_prefill(
        self, continuation: KVRetentionLayerContinuation,
    ) -> TTFTLayerwisePrefillBoundary:
        """Advance explicit worker state; no generator retains phase views."""
        state = continuation.state
        execution = continuation.execution
        if state is None or state.end_layer != self.end_layer:
            raise RuntimeError("invalid explicit Prefill model state")
        submission_events = []
        while state.next_layer < state.end_layer:
            group_end_layer = min(
                state.next_layer + execution.group_layers, state.end_layer)
            group_start_ts = time.perf_counter()
            state.hidden_states, state.residual = self._forward_decoder_layer_range(
                start_layer=state.next_layer,
                end_layer=group_end_layer,
                positions=state.positions,
                hidden_states=state.hidden_states,
                residual=state.residual,
                aux_hidden_states=state.aux_hidden_states,
                workspace_active=state.workspace_active,
            )
            execution.record_group(
                start_layer=state.next_layer, end_layer=group_end_layer,
                host_wall_s=time.perf_counter() - group_start_ts)
            state.next_layer = group_end_layer
            if (state.next_layer < state.end_layer
                    and getattr(self, "isolation_p1_boundary_fastpath", False)
                    and execution.consume_yield_request()):
                # The runner fences the latest group before exporting a frame
                # or releasing pages. Avoid an extra older-group helper wait.
                return execution.make_boundary(
                    next_layer=state.next_layer, end_layer=state.end_layer)
            if (state.next_layer < state.end_layer
                    and getattr(self, "kv_retention_p1_parked_enabled", False)):
                kv_retention_bound_prefill_submission(
                    submission_events, completed_fastpath=getattr(
                        self, "isolation_p1_boundary_fastpath", False))
            if (state.next_layer < state.end_layer
                    and execution.consume_yield_request()):
                return execution.make_boundary(
                    next_layer=state.next_layer, end_layer=state.end_layer)
        if not get_pp_group().is_last_rank:
            result = IntermediateTensors({"hidden_states": state.hidden_states,
                                          "residual": state.residual})
        else:
            hidden_states, _ = self.norm(state.hidden_states, state.residual)
            result = ((hidden_states, state.aux_hidden_states)
                      if state.aux_hidden_states else hidden_states)
        continuation.close()
        raise StopIteration(result)

    def _ttft_resumable_prefill_forward(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        aux_hidden_states: list[torch.Tensor],
        workspace_active: bool,
        execution: TTFTLayerwisePrefillExecution,
    ) -> Generator[
        TTFTLayerwisePrefillBoundary,
        None,
        torch.Tensor
        | IntermediateTensors
        | tuple[torch.Tensor, list[torch.Tensor]],
    ]:
        """Run one bounded layer group per ``next()`` call.

        The generator and every tensor it closes over stay inside one
        GPUModelRunner Prefill slot.  Only the tensor-free boundary yielded by
        this generator may be returned to EngineCore.
        """

        next_layer = self.start_layer
        try:
            while next_layer < self.end_layer:
                group_end_layer = min(
                    next_layer + execution.group_layers,
                    self.end_layer,
                )
                group_start_ts = time.perf_counter()
                hidden_states, residual = self._forward_decoder_layer_range(
                    start_layer=next_layer,
                    end_layer=group_end_layer,
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    aux_hidden_states=aux_hidden_states,
                    workspace_active=workspace_active,
                )
                execution.record_group(
                    start_layer=next_layer,
                    end_layer=group_end_layer,
                    host_wall_s=time.perf_counter() - group_start_ts,
                )
                next_layer = group_end_layer
                if (
                    next_layer < self.end_layer
                    and execution.consume_yield_request()
                ):
                    yield execution.make_boundary(
                        next_layer=next_layer,
                        end_layer=self.end_layer,
                    )
        finally:
            execution.cancel_yield_request()

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> (
        torch.Tensor
        | IntermediateTensors
        | tuple[torch.Tensor, list[torch.Tensor]]
        | TTFTLayerwisePrefillModelYield
    ):
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        workspace_active = False
        if self.pd_prefill_workspace_enabled:
            workspace_active = prepare_active_llama_prefill_workspace(
                num_tokens=int(hidden_states.shape[0]),
                hidden_size=int(self.config.hidden_size),
                intermediate_size=self.pd_prefill_workspace_intermediate_size,
                qkv_width=self.pd_prefill_workspace_qkv_width,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
                tensor_parallel_size=self.pd_prefill_workspace_tp_size,
                quantized=self.quant_config is not None,
                lora_enabled=self.pd_prefill_workspace_lora_enabled,
                opmux_enabled=self.pd_prefill_workspace_opmux_enabled,
                mixed_phase_layer_enabled=(
                    self.pd_prefill_workspace_mixed_phase_enabled
                ),
                layer_path_supported=(
                    self.pd_prefill_workspace_layer_path_supported
                ),
            )

        aux_hidden_states = []
        ttft_layer_execution = (
            active_ttft_layerwise_prefill_execution()
            if self.ttft_layerwise_prefill_execution_enabled
            else None
        )
        if ttft_layer_execution is None:
            # Exact rollback path: keep the previous whole-model Python loop
            # unchanged when the new experiment flag is disabled.
            for idx, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)
            ):
                if idx in self.aux_hidden_state_layers:
                    aux_hidden_states.append(hidden_states + residual)
                if workspace_active:
                    hidden_states, residual = (
                        layer.forward_with_pd_prefill_workspace(
                            positions,
                            hidden_states,
                            residual,
                        )
                    )
                else:
                    hidden_states, residual = layer(
                        positions,
                        hidden_states,
                        residual,
                    )
                if self.pd_layerwise_profile_handoff_enabled:
                    pd_layerwise_handoff_layer_boundary(
                        self.start_layer + idx
                    )
        elif ttft_layer_execution.preemption_active:
            if kv_retention_resume_frame_requested():
                continuation = KVRetentionLayerContinuation(
                    self, ttft_layer_execution,
                    KVRetentionModelState(
                        self.start_layer, self.end_layer, positions, hidden_states,
                        residual, aux_hidden_states, workspace_active))
            else:
                continuation = self._ttft_resumable_prefill_forward(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    aux_hidden_states=aux_hidden_states,
                    workspace_active=workspace_active,
                    execution=ttft_layer_execution,
                )
            try:
                boundary = next(continuation)
            except StopIteration as completed:
                return completed.value
            return TTFTLayerwisePrefillModelYield(
                boundary=boundary,
                continuation=continuation,
            )
        else:
            next_layer = self.start_layer
            while next_layer < self.end_layer:
                group_end_layer = min(
                    next_layer + ttft_layer_execution.group_layers,
                    self.end_layer,
                )
                group_start_ts = time.perf_counter()
                hidden_states, residual = self._forward_decoder_layer_range(
                    start_layer=next_layer,
                    end_layer=group_end_layer,
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    aux_hidden_states=aux_hidden_states,
                    workspace_active=workspace_active,
                )
                ttft_layer_execution.record_group(
                    start_layer=next_layer,
                    end_layer=group_end_layer,
                    host_wall_s=time.perf_counter() - group_start_ts,
                )
                next_layer = group_end_layer

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class LlamaForCausalLM(
    nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    # Mistral/Llama models can also be loaded with --load-format mistral
    # from consolidated.safetensors checkpoints
    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "qscale_act": "input_scale",
        "qscale_weight": "weight_scale",
        "kv_fake_quantizer.qscale_act": "kv_scale",
        "q_fake_quantizer.qscale_act": "attn.q_scale",
        "k_fake_quantizer.qscale_act": "k_scale",
        "v_fake_quantizer.qscale_act": "v_scale",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm",
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.model = self._init_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """Override to return default layers for Llama

        Note: The GPU model runner will override this with layers from
        the speculative config if available, providing dynamic configuration.
        """
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = LlamaDecoderLayer,
    ):
        return LlamaModel(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(
            self.maybe_remap_mistral(name, loaded_weight)
            for name, loaded_weight in weights
        )

    # This function is used to remap the mistral format as
    # used by Mistral and Llama <=2
    def maybe_remap_mistral(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> tuple[str, torch.Tensor]:
        def permute(w: torch.Tensor, n_heads: int, attn_out: int):
            attn_in = self.config.head_dim * n_heads

            return (
                w.view(n_heads, attn_in // n_heads // 2, 2, attn_out)
                .transpose(1, 2)
                .reshape(attn_in, attn_out)
            )

        mapping = self.mistral_mapping
        modules = name.split(".")

        # rotary embeds should be sliced
        # If using quantized model in mistral format,
        # quantization scales (qscale_weight) also need to be sliced
        if "wk" in modules and modules[-1] == "weight":
            loaded_weight = permute(
                loaded_weight, self.config.num_key_value_heads, self.config.hidden_size
            )
        elif (
            "wk" in modules
            and modules[-1] == "qscale_weight"
            and loaded_weight.numel() > 1
        ):
            loaded_weight = permute(loaded_weight, self.config.num_key_value_heads, 1)
        elif "wq" in modules and modules[-1] == "weight":
            loaded_weight = permute(
                loaded_weight, self.config.num_attention_heads, self.config.hidden_size
            )
        elif (
            "wq" in modules
            and modules[-1] == "qscale_weight"
            and loaded_weight.numel() > 1
        ):
            loaded_weight = permute(loaded_weight, self.config.num_attention_heads, 1)

        num_modules = len(modules)
        for i in range(num_modules):
            item = modules[i]
            next_item = modules[i + 1] if i < num_modules - 1 else None

            combined_item = f"{item}.{next_item}" if next_item is not None else None

            if combined_item in mapping:
                name = name.replace(combined_item, mapping[combined_item])
            elif item in mapping and mapping[item] not in name:
                name = name.replace(item, mapping[item])

        return name, loaded_weight


class LlamaBidirectionalForSequenceClassification(as_seq_cls_model(LlamaForCausalLM)):
    # This class sets the correct attention type and pooling type
    # through LlamaBidirectionalConfig.
    pass


class LlamaBidirectionalModel(as_embedding_model(LlamaForCausalLM)):
    # This class sets the correct attention type and pooling type
    # through LlamaBidirectionalConfig.
    pass
