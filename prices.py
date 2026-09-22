"""Gemini API list prices, in US dollars per million tokens (paid tier).

Source: https://ai.google.dev/gemini-api/docs/pricing   Checked: 2026-09-21
The output price includes thinking tokens. gemini-3.8-flash prices are valid through
2026-12-31 and double on 2027-01-01 (input $1.50, output $7.50). Even on the free tier
(you pay $0) this lab prices everything at these list prices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

PRICE_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"
PRICE_CHECKED = "2026-09-21"

#: Batch API requests cost half the standard price.
BATCH_DISCOUNT = 0.50

#: A cached-input read costs 10% of the normal input price (storage is billed separately).
CACHE_READ_FRACTION = {
    "gemini-3.1-flash-lite": 0.10,
    "gemini-3.5-flash-lite": 0.10,
    "gemini-3.8-flash": 0.10,
    "gemini-3.1-pro-preview": 0.10,
}


@dataclass(frozen=True)
class Model:
    model_id: str          # exact string sent as `model` in the API request
    input_per_mtok: float  # USD per 1M input tokens
    output_per_mtok: float # USD per 1M output tokens (thinking included)


MODELS: Dict[str, Model] = {
    "gemini-3.1-flash-lite": Model("gemini-3.1-flash-lite", 0.25, 1.50),
    "gemini-3.5-flash-lite": Model("gemini-3.5-flash-lite", 0.30, 2.50),
    "gemini-3.8-flash": Model("gemini-3.8-flash", 0.75, 3.75),
    "gemini-3.1-pro-preview": Model("gemini-3.1-pro-preview", 2.00, 12.00),
}
DEFAULT_MODEL = "gemini-3.8-flash"


def cost_usd(model_key: str, input_tokens: int, output_tokens: int) -> float:
    """List-price cost of one request in USD (no caching, no batch discount, no free tier)."""
    m = MODELS[model_key]
    return (input_tokens * m.input_per_mtok + output_tokens * m.output_per_mtok) / 1_000_000
