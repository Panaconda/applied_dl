from __future__ import annotations

from typing import Dict

CLASS_PROMPTS: Dict[str, str] = {
    "Pneumonia": (
        "Findings: Frontal radiograph of a child. "
        "Evaluation reveals opacities. "
        "Impressions: pneumonia."
    ),
    "Bronchiolitis": (
        "Findings: Frontal radiograph of a child. "
        "Evaluation reveals reticulonodular opacities. "
        "Impressions: bronchiolitis."
    ),
    "Bronchitis": (
        "Findings: Frontal radiograph of a child. "
        "Evaluation reveals bronchial wall thickening. "
        "Impressions: bronchitis."
    ),
    "Brocho-pneumonia": (
        "Findings: Frontal radiograph of a child. "
        "Evaluation reveals patchy opacities. "
        "Impressions: brocho-pneumonia."
    ),
}
