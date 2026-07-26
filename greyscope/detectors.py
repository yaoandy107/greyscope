"""Pinned open-detector adapters used by the release comparison harness."""
from __future__ import annotations

import re
from html import unescape
from pathlib import Path
from typing import Callable

import numpy as np


def fakespot_clean_text(text: str) -> str:
    """The preprocessing published with Fakespot's detector, reproduced verbatim in effect."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\(.*?\)", r"\1", text)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    text = re.sub(r"#+ ", "", text)
    text = re.sub(r"^>.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^(\s*[-*+]|\d+\.)\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\|.*?\|", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = unescape(text)
    text = text.replace("\n", " ").replace("\t", " ").replace("^M", " ").replace("\r", " ")
    text = text.replace(" ,", ",")
    return re.sub(" +", " ", text)


def scores_from_logits(logits, *, ai_label_id: int | None = None) -> np.ndarray:
    """Convert classifier logits to a continuous score, higher = more AI involvement."""
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expected [rows, labels] logits, got {values.shape}")
    if values.shape[1] == 1:
        return 1.0 / (1.0 + np.exp(-values[:, 0]))
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    if ai_label_id is not None:
        if not 0 <= ai_label_id < values.shape[1]:
            raise ValueError(f"ai_label_id {ai_label_id} outside {values.shape[1]} labels")
        return probabilities[:, ai_label_id]
    ranks = np.arange(values.shape[1], dtype=float) / (values.shape[1] - 1)
    return probabilities @ ranks


def graded_scores_from_logits(logits, *, head_type: str) -> np.ndarray:
    """Decode Greyscope's v1 softmax head or v2 CORN head to expected rank."""
    if head_type == "seqcls":
        return scores_from_logits(logits)
    if head_type == "corn":
        from greyscope.corn import corn_scalar_score

        return corn_scalar_score(logits)
    raise ValueError(f"unknown Greyscope head type: {head_type}")


def make_transformers_scorer(
    spec: dict,
    *,
    device: str = "cuda",
    batch_size: int = 32,
) -> tuple[Callable[[list[str]], np.ndarray], object, object]:
    """Load a pinned Transformers/Fakespot/Desklib detector and return its score function."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    source = spec["source"]
    revision = spec["revision"]
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    if spec["adapter"] == "desklib":
        from torch import nn
        from transformers import PreTrainedModel

        class DesklibAIDetectionModel(PreTrainedModel):
            config_class = AutoConfig

            def __init__(self, config):
                super().__init__(config)
                self.model = AutoModel.from_config(config)
                self.classifier = nn.Linear(config.hidden_size, 1)
                self.post_init()

            def forward(self, input_ids, attention_mask=None, **kwargs):
                hidden = self.model(
                    input_ids, attention_mask=attention_mask, **kwargs
                )[0]
                expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                pooled = torch.sum(hidden * expanded, dim=1) / torch.clamp(
                    expanded.sum(dim=1), min=1e-9
                )
                return {"logits": self.classifier(pooled)}

        model = DesklibAIDetectionModel.from_pretrained(source, revision=revision)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(source, revision=revision)
    model = model.eval().to(device)
    preprocess = fakespot_clean_text if spec["adapter"] == "fakespot" else str

    def score(texts: list[str]) -> np.ndarray:
        rows = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = [preprocess(str(text)) for text in texts[start : start + batch_size]]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=spec["max_length"],
                    return_tensors="pt",
                ).to(device)
                output = model(**encoded)
                logits = output["logits"] if isinstance(output, dict) else output.logits
                rows.append(logits.float().cpu().numpy())
        return scores_from_logits(
            np.concatenate(rows), ai_label_id=spec.get("ai_label_id")
        )

    return score, tokenizer, model


def make_greyscope_scorer(
    spec: dict,
    *,
    device: str = "cuda",
    batch_size: int = 16,
) -> tuple[Callable[[list[str]], np.ndarray], object, object]:
    """Load a pinned Greyscope artifact and reproduce its calibrated scalar score."""
    import json

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from greyscope.preprocess import clean_text

    source = spec["source"]
    revision = spec["revision"]
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    load_kwargs = {"revision": revision, "dtype": torch.bfloat16}
    if spec["id"].endswith("int4"):
        load_kwargs["device_map"] = device
    model = AutoModelForSequenceClassification.from_pretrained(
        source, **load_kwargs
    ).eval()
    if not spec["id"].endswith("int4"):
        model = model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    calibration_path = hf_hub_download(
        source, "calibration.json", revision=revision
    )
    calibration = json.loads(Path(calibration_path).read_text())
    model.greyscope_calibration = calibration

    def score(texts: list[str]) -> np.ndarray:
        values = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                prompts = [
                    calibration["prompt_template"].format(
                        text=clean_text(
                            str(text),
                            normalize=spec.get("normalize_unicode", True),
                        )
                    )
                    for text in texts[start : start + batch_size]
                ]
                encoded = tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=spec["max_length"],
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to(device)
                logits = model(**encoded).logits.float().cpu().numpy()
                values.append(
                    graded_scores_from_logits(logits, head_type=spec["head_type"])
                )
        raw = np.concatenate(values)
        oriented = -raw if calibration["flip"] else raw
        scaled = (oriented - calibration["score_min"]) / (
            calibration["score_max"] - calibration["score_min"]
        )
        return np.clip(scaled, 0.0, 1.0)

    return score, tokenizer, model


def make_editlens_llama_scorer(
    spec: dict,
    *,
    device: str = "cuda",
    batch_size: int = 16,
) -> tuple[Callable[[list[str]], np.ndarray], object, object]:
    """Reconstruct EditLens's pinned normalized four-class Llama adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    class NormedLinear(torch.nn.Module):
        def __init__(self, hidden: int, labels: int):
            super().__init__()
            self.norm = torch.nn.LayerNorm(hidden)
            self.linear = torch.nn.Linear(hidden, labels, bias=False)

        def forward(self, values):
            return self.linear(self.norm(values))

    tokenizer = AutoTokenizer.from_pretrained(
        spec["source"], revision=spec["revision"]
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        spec["base_source"],
        revision=spec["base_revision"],
        num_labels=4,
        dtype=torch.bfloat16,
    )
    model.score = NormedLinear(model.config.hidden_size, 4).to(torch.bfloat16)
    model = PeftModel.from_pretrained(
        model, spec["source"], revision=spec["revision"]
    ).eval().to(device)
    model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def score(texts: list[str]) -> np.ndarray:
        values = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    texts[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=spec["max_length"],
                    return_tensors="pt",
                ).to(device)
                values.append(model(**encoded).logits.float().cpu().numpy())
        return scores_from_logits(np.concatenate(values))

    return score, tokenizer, model


def make_meld_scorer(
    spec: dict,
    *,
    device: str = "cuda",
    batch_size: int = 16,
) -> tuple[Callable[[list[str]], np.ndarray], object, object]:
    """Load MELD's pinned release and reproduce its overlapping-chunk scoring."""
    import json

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    class MELDDetector(nn.Module):
        def __init__(self, config: dict):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(
                spec["base_source"],
                revision=spec["base_revision"],
                attn_implementation="sdpa",
            )
            if hasattr(self.backbone.config, "reference_compile"):
                self.backbone.config.reference_compile = False
            hidden = self.backbone.config.hidden_size
            dropout = config.get("dropout", 0.1)
            self.dropout = nn.Dropout(dropout)
            self.head_main = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, config.get("num_labels", 2)),
            )
            self.head_gen = nn.Linear(hidden, config["n_generators"])
            self.head_att = nn.Linear(hidden, config["n_attacks"])
            self.head_dom = nn.Linear(hidden, config["n_domains"])
            self.log_var_main = nn.Parameter(torch.zeros(()))
            self.log_var_gen = nn.Parameter(torch.zeros(()))
            self.log_var_att = nn.Parameter(torch.zeros(()))
            self.log_var_dom = nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, attention_mask):
            hidden = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            return self.head_main(self.dropout(pooled)).float()

    source = spec["source"]
    revision = spec["revision"]
    config_path = hf_hub_download(source, "meld_config.json", revision=revision)
    config = json.loads(Path(config_path).read_text())
    if config["max_length"] != spec["max_length"]:
        raise ValueError(
            f"MELD max_length drifted: manifest={spec['max_length']} "
            f"checkpoint={config['max_length']}"
        )
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    model = MELDDetector(config).to(device)
    weights_path = hf_hub_download(source, "model.safetensors", revision=revision)
    model.load_state_dict(load_file(weights_path), strict=True)
    model = model.eval()

    def score(texts: list[str]) -> np.ndarray:
        encoded = tokenizer(
            [str(text) for text in texts],
            truncation=True,
            max_length=spec["max_length"],
            stride=min(512, spec["max_length"] - 1),
            return_overflowing_tokens=True,
        )
        sample_ids = np.asarray(encoded.pop("overflow_to_sample_mapping"), dtype=int)
        chunk_scores = []
        with torch.no_grad():
            for start in range(0, len(sample_ids), batch_size):
                batch = tokenizer.pad(
                    {
                        "input_ids": encoded["input_ids"][start : start + batch_size],
                        "attention_mask": encoded["attention_mask"][
                            start : start + batch_size
                        ],
                    },
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                chunk_scores.extend(
                    torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
                )
        sums = np.bincount(
            sample_ids,
            weights=np.asarray(chunk_scores, dtype=float),
            minlength=len(texts),
        )
        counts = np.bincount(sample_ids, minlength=len(texts))
        if np.any(counts == 0):
            raise ValueError("MELD tokenizer produced no chunks for an input")
        return sums / counts

    return score, tokenizer, model


def make_binoculars_scorer(
    spec: dict,
    *,
    device: str = "cuda",
    batch_size: int = 1,
) -> tuple[Callable[[list[str]], np.ndarray], object, object]:
    """Reproduce pinned Binoculars scoring; returned scores are oriented higher = AI."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        spec["observer_source"], revision=spec["observer_revision"]
    )
    performer_tokenizer = AutoTokenizer.from_pretrained(
        spec["performer_source"], revision=spec["performer_revision"]
    )
    if tokenizer.get_vocab() != performer_tokenizer.get_vocab():
        raise ValueError("Binoculars observer and performer tokenizers differ")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": False,
    }
    observer = AutoModelForCausalLM.from_pretrained(
        spec["observer_source"],
        revision=spec["observer_revision"],
        **load_kwargs,
    ).eval().to(device)
    performer = AutoModelForCausalLM.from_pretrained(
        spec["performer_source"],
        revision=spec["performer_revision"],
        **load_kwargs,
    ).eval().to(device)

    def score(texts: list[str]) -> np.ndarray:
        rows = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    texts[start : start + batch_size],
                    padding="longest",
                    truncation=True,
                    max_length=spec["max_length"],
                    return_token_type_ids=False,
                    return_tensors="pt",
                ).to(device)
                observer_logits = observer(**encoded).logits
                performer_logits = performer(**encoded).logits

                shifted_logits = performer_logits[..., :-1, :].contiguous()
                shifted_labels = encoded.input_ids[..., 1:].contiguous()
                shifted_mask = encoded.attention_mask[..., 1:].to(shifted_logits.dtype)
                token_loss = F.cross_entropy(
                    shifted_logits.transpose(1, 2),
                    shifted_labels,
                    reduction="none",
                )
                perplexity = (token_loss * shifted_mask).sum(1) / shifted_mask.sum(1)

                observer_probs = torch.softmax(observer_logits, dim=-1)
                cross_entropy = -(observer_probs * torch.log_softmax(
                    performer_logits, dim=-1
                )).sum(-1)
                mask = encoded.attention_mask.to(cross_entropy.dtype)
                entropy = (cross_entropy * mask).sum(1) / mask.sum(1)
                rows.extend((-(perplexity / entropy)).float().cpu().tolist())
        return np.asarray(rows, dtype=float)

    return score, tokenizer, (observer, performer)
