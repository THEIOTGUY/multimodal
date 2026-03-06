"""
StreamVLNWithValueHead — Actor-Critic wrapper for StreamVLN.

Wraps ``StreamVLNForCausalLM`` with a scalar value head (critic) so that
a single forward pass returns ``(lm_logits, loss, value)`` — the 3-tuple
expected by the MindDrive PPO pipeline.

Architecture (from the MindDrive diagram, Block 1):
┌─────────────────────────────────────────────────────┐
│  StreamVLN Base (Actor) + Value Head (Critic)       │
│                                                     │
│  Video + Text Prompt ──► Token Probs (Action)       │
│           Actor: Predicts Next Token                │
│                        ──► Value Head (Critic)      │
│                            Expected Future Reward   │
└─────────────────────────────────────────────────────┘

Block 2 (Initialization / KL-Penalty):
┌─────────────────────────────────────────────────────┐
│  Pre-trained IL Weights → LoRA Adapter              │
│  Frozen Model (99% Weights) + RL Model (LoRA)       │
│  Reference Model (Frozen StreamVLN Copy)            │
│  KL-Penalty prevents catastrophic forgetting        │
└─────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional, List, Tuple


class ValueHead(nn.Module):
    """
    Scalar value head that maps hidden states → per-token value estimates.

    Architecture:
        LayerNorm → Dropout → Linear(hidden_size → 1)

    This is a standalone implementation so we do not depend on the ``trl``
    library's ``ValueHead`` at import time (trl may be vendored or absent).
    """

    def __init__(self, config, dropout_prob: float = 0.0):
        super().__init__()
        hidden_size = config.hidden_size
        self.summary = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : Tensor  [B, seq_len, hidden_size]

        Returns
        -------
        values : Tensor  [B, seq_len, 1]
        """
        x = self.layer_norm(hidden_states)
        x = self.dropout(x)
        return self.summary(x)


class StreamVLNWithValueHead(nn.Module):
    """
    Wraps ``StreamVLNForCausalLM`` with a learned value head (Linear → scalar).

    The wrapper is designed to be a drop-in replacement inside the MindDrive
    PPO pipeline: ``.forward()`` → ``(logits, loss, value)`` and
    ``.generate()`` → delegated to the underlying StreamVLN model (with
    streaming KV cache support).

    Parameters
    ----------
    pretrained_model : StreamVLNForCausalLM
        The pre-trained (IL) StreamVLN model.
    summary_dropout_prob : float, optional
        Dropout probability inside the ValueHead (default 0.0).
    v_head_init_strategy : str or None, optional
        ``"normal"`` for Gaussian init, ``None`` for PyTorch default.
    v_head_initializer_range : float, optional
        Std-dev for Gaussian init (default 0.2).
    """

    def __init__(
        self,
        pretrained_model,
        summary_dropout_prob: float = 0.0,
        v_head_init_strategy: Optional[str] = "normal",
        v_head_initializer_range: float = 0.2,
    ):
        super().__init__()
        self.pretrained_model = pretrained_model

        # Track PEFT status — updated after LoRA application in train_rl.py
        self.is_peft_model = False

        # Attach the value head (critic): maps hidden_size → 1
        self.v_head = ValueHead(
            self.pretrained_model.config,
            dropout_prob=summary_dropout_prob,
        )
        self._init_weights(v_head_init_strategy, v_head_initializer_range)
        # Keep critic dtype aligned with the base model to avoid bf16/fp32
        # mismatches during layer norm / linear ops.
        self.v_head = self.v_head.to(dtype=self.pretrained_model.dtype)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, strategy: Optional[str], init_range: float):
        if strategy == "normal":
            self.v_head.summary.weight.data.normal_(mean=0.0, std=init_range)
            self.v_head.summary.bias.data.zero_()

    # ------------------------------------------------------------------
    # Forward  (Actor + Critic)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # StreamVLN multimodal inputs
        images: Optional[List] = None,
        depths: Optional[torch.FloatTensor] = None,
        poses: Optional[torch.FloatTensor] = None,
        intrinsics: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values=None,
        # Extra kwargs forwarded to StreamVLN
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[torch.Tensor], torch.FloatTensor]:
        """
        Returns
        -------
        lm_logits : FloatTensor  [B, seq_len, vocab_size]
            Language-model logits (upcast to fp32).
        loss : Tensor or None
            Cross-entropy loss if ``labels`` is provided.
        value : FloatTensor  [B, seq_len]
            Per-token scalar value prediction from the critic.
        """
        # Force hidden state output for value head
        kwargs["output_hidden_states"] = True

        base_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            depths=depths,
            poses=poses,
            intrinsics=intrinsics,
            labels=labels,
            past_key_values=past_key_values,
            **kwargs,
        )

        # Extract components
        lm_logits = base_output.logits
        loss = base_output.loss
        last_hidden_state = base_output.hidden_states[-1]

        # Align hidden state with value-head device/dtype.
        v_head_param = self.v_head.summary.weight
        if (
            last_hidden_state.device != v_head_param.device
            or last_hidden_state.dtype != v_head_param.dtype
        ):
            last_hidden_state = last_hidden_state.to(
                device=v_head_param.device,
                dtype=v_head_param.dtype,
            )

        # Critic: hidden_state → scalar per token
        value = self.v_head(last_hidden_state).squeeze(-1)  # [B, seq_len]

        # Upcast logits to fp32 for numerical stability
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        return (lm_logits, loss, value)

    # ------------------------------------------------------------------
    # Generate  (delegates to StreamVLN's streaming generation)
    # ------------------------------------------------------------------

    def generate(self, *args, **kwargs):
        """
        Delegates to ``StreamVLNForCausalLM.generate()`` — no value head
        is needed during action generation at inference / rollout time.
        """
        return self.pretrained_model.generate(*args, **kwargs)

    # ------------------------------------------------------------------
    # Streaming state management (passthrough to base model)
    # ------------------------------------------------------------------

    def reset(self, env_num: int):
        """Reset streaming KV caches for all environments."""
        self.pretrained_model.reset(env_num)

    def reset_for_env(self, env_idx: int):
        """Reset streaming KV cache for a single environment."""
        self.pretrained_model.reset_for_env(env_idx)

    # ------------------------------------------------------------------
    # State dict (save/load)
    # ------------------------------------------------------------------

    def state_dict(self, *args, **kwargs):
        """
        Returns state dict.  When using PEFT (LoRA), only the
        value-head weights and LoRA adapters are saved.
        """
        if self.is_peft_model:
            pretrained_state = {}
        else:
            pretrained_state = self.pretrained_model.state_dict(*args, **kwargs)

        v_head_state = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state.items():
            pretrained_state[f"v_head.{k}"] = v
        return pretrained_state

    def load_v_head(self, state_dict):
        """Load value-head weights from a saved state dict."""
        v_head_state = {}
        for k in list(state_dict.keys()):
            if "v_head." in k:
                v_head_state[k.replace("v_head.", "")] = state_dict.pop(k)
        if v_head_state:
            self.v_head.load_state_dict(v_head_state, strict=False)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def device(self):
        return next(self.pretrained_model.parameters()).device

    @property
    def dtype(self):
        return next(self.pretrained_model.parameters()).dtype

    @property
    def config(self):
        return self.pretrained_model.config

    def train(self, mode: bool = True):
        """Set training mode — value head always trains, base model follows mode."""
        self.v_head.train(mode)
        self.pretrained_model.train(mode)
        # Set the top-level module flag without recursing (already handled above)
        nn.Module.train(self, mode)
        return self

    def eval(self):
        """Set eval mode."""
        return self.train(False)


def create_reference_model(
    model: StreamVLNWithValueHead,
) -> Optional[StreamVLNWithValueHead]:
    """
    Create a frozen copy of the model to serve as the reference
    policy for KL-penalty computation (Block 2 of MindDrive).

    With LoRA, the reference model is simply the base model with
    LoRA adapters disabled (no copy needed — handled by PEFT's
    ``disable_adapter()`` context manager). For full-parameter
    training, we deep-copy the entire model and freeze it.

    Returns
    -------
    StreamVLNWithValueHead or None
        None when LoRA is active (ref logits via disable_adapter).
    """
    if hasattr(model, "is_peft_model") and model.is_peft_model:
        # With PEFT/LoRA, TRL can use disable_adapter() context manager
        # No need to duplicate the model — memory efficient
        return None
    else:
        ref_model = deepcopy(model)
        ref_model.requires_grad_(False)
        ref_model.eval()
        return ref_model
