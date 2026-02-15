from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import torch


class InferenceEngine:
    def __init__(self) -> None:
        self.logger = logging.getLogger("ciclopes.inference_engine")
        self.initialized_at = datetime.now(timezone.utc)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.logger.info("Initializing InferenceEngine on device=%s", self.device)

    def forward(self, x: Any) -> dict[str, Any]:
        if isinstance(x, list) and all(isinstance(item, torch.Tensor) for item in x):
            moved_batches = [tensor.to(self.device) for tensor in x]
            return {"test": moved_batches}

        if isinstance(x, torch.Tensor):
            return {"test": x.to(self.device)}

        return {"test": x}

    def status(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "initialized_at": self.initialized_at.isoformat(),
            "cuda_available": bool(torch.cuda.is_available()),
        }
