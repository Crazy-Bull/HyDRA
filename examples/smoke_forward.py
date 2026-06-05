from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hydra_rffi import HyDRA, max_softmax_decision


def main() -> None:
    batch_size = 2
    input_length = 256
    input_channels = 6
    num_classes = 28

    x = torch.randn(batch_size, input_length, input_channels)

    tdse = HyDRA(mode="tdse", input_channels=input_channels, num_classes=num_classes)
    mlfe = HyDRA(mode="mlfe", input_channels=input_channels, num_classes=num_classes)

    tdse_logits = tdse(x)
    mlfe_logits = mlfe(x)

    print("TDSE logits:", tuple(tdse_logits.shape))
    print("MLFE logits:", tuple(mlfe_logits.shape))

    decision = max_softmax_decision(tdse_logits, threshold=0.999, temperature=1.6)
    print("Open-set labels:", decision.labels)


if __name__ == "__main__":
    main()
