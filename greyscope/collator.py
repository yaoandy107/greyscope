"""Collator for the seq-cls head: tokenizes the prompt + integer bucket label.

Right padding is mandatory: the head reads the last non-pad token, so left
padding would point it at padding instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DataCollatorForSeqCls:
    tokenizer: Any
    max_length: int = 2048
    pad_to_multiple_of: int | None = 8

    def __post_init__(self) -> None:
        if self.tokenizer.padding_side != "right":
            raise ValueError(
                f"DataCollatorForSeqCls requires tokenizer.padding_side='right', "
                f"got {self.tokenizer.padding_side!r}."
            )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        texts = [ex["prompt"] for ex in examples]
        # add_special_tokens=False: keep the prompt's final "Answer:" token as the
        # last non-pad token (what the seq-cls head reads); an auto-appended special
        # token would displace it.
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
            add_special_tokens=False,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor([int(ex["label"]) for ex in examples], dtype=torch.long),
        }
