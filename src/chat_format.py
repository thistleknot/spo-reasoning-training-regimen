"""
Chat-format helpers for Qwen instruction tuning and inference.

The local ablation runner trains an instruct model. Training and inference must
therefore use the same chat-turn surface rather than raw text continuation.
"""

import re
from typing import Any


THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def build_generation_prompt(tokenizer: Any, user_text: str) -> str:
    """Build the inference-time chat prompt for a single user turn.

    Preconditions:
        `user_text` is the full task prompt the model should answer.
    Failure modes:
        Falls back to raw text when the tokenizer does not expose a chat
        template.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_text


def build_training_conversation(tokenizer: Any, user_text: str, assistant_text: str) -> str:
    """Build the full supervised chat exchange used for SFT labels."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    conversation = f"{user_text}\n\n{assistant_text}"
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token:
        conversation = f"{conversation}{eos_token}"
    return conversation


def strip_response_preamble(text: str) -> str:
    """Remove Qwen thinking scaffolding from a decoded assistant response."""
    return THINK_BLOCK_RE.sub("", text.strip(), count=1).strip()
