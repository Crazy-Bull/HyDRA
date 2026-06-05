from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class OpenSetDecision:
    labels: list[str]
    class_indices: torch.Tensor
    max_probabilities: torch.Tensor
    probabilities: torch.Tensor


def temperature_softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    return torch.softmax(logits / temperature, dim=-1)


def max_softmax_decision(
    logits: torch.Tensor,
    class_names: Optional[Sequence[str]] = None,
    threshold: float = 0.999,
    temperature: float = 1.6,
    illegal_label: str = "illegal",
) -> OpenSetDecision:
    """Classify known transmitters or reject them using max softmax probability."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")

    probabilities = temperature_softmax(logits, temperature=temperature)
    max_probabilities, class_indices = probabilities.max(dim=-1)

    if class_names is None:
        class_names = [str(i) for i in range(logits.size(-1))]
    if len(class_names) != logits.size(-1):
        raise ValueError("class_names must match the number of logit classes.")

    labels = []
    for prob, index in zip(max_probabilities.tolist(), class_indices.tolist()):
        labels.append(class_names[index] if prob >= threshold else illegal_label)

    return OpenSetDecision(
        labels=labels,
        class_indices=class_indices,
        max_probabilities=max_probabilities,
        probabilities=probabilities,
    )
