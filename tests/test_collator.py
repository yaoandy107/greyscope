"""Tests for the sequence-classification collator."""

import pytest

torch = pytest.importorskip("torch")


class _FakeTokenizer:
    """Minimal tokenizer stand-in for collator unit tests."""

    pad_token_id = 0
    eos_token_id = 9
    pad_token = "[PAD]"
    eos_token = "</s>"
    padding_side = "right"

    def __call__(self, texts, padding, truncation, max_length, return_tensors,
                 pad_to_multiple_of, add_special_tokens=True):
        # toy "BPE": each char → its ord(). Pad with 0 to a multiple of pad_to_multiple_of.
        seqs = [[ord(c) for c in t][:max_length] for t in texts]
        max_len = max(len(s) for s in seqs)
        if pad_to_multiple_of:
            max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        input_ids = torch.tensor([s + [self.pad_token_id] * (max_len - len(s)) for s in seqs])
        attention_mask = (input_ids != self.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_seqcls_collator_builds_integer_labels():
    from greyscope.collator import DataCollatorForSeqCls

    tok = _FakeTokenizer()
    collator = DataCollatorForSeqCls(tokenizer=tok, max_length=64, pad_to_multiple_of=4)
    batch = collator([
        {"prompt": "Hello world", "label": 1},
        {"prompt": "Hi", "label": 3},
    ])

    assert batch["labels"].tolist() == [1, 3]
    assert batch["labels"].dtype == torch.long
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["attention_mask"].shape


def test_seqcls_collator_pads_to_multiple():
    from greyscope.collator import DataCollatorForSeqCls

    tok = _FakeTokenizer()
    collator = DataCollatorForSeqCls(tokenizer=tok, max_length=64, pad_to_multiple_of=8)
    batch = collator([{"prompt": "abc", "label": 0}])
    assert batch["input_ids"].shape[1] % 8 == 0


def test_seqcls_collator_rejects_left_padding():
    """Left padding would point the seq-cls head at padding, not the last real token.

    With left padding the classification position (derived from pad-token
    locations) lands in the padding region. We catch this at construction time.
    """
    from greyscope.collator import DataCollatorForSeqCls

    tok = _FakeTokenizer()
    tok.padding_side = "left"

    with pytest.raises(ValueError, match="padding_side='right'"):
        DataCollatorForSeqCls(tokenizer=tok, max_length=64, pad_to_multiple_of=8)
