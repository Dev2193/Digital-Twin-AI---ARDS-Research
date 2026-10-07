"""BioClinicalBERT builder for criterion-level report classification."""

from __future__ import annotations

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install the NLP extra: pip install -e '.[nlp]'") from exc

BIOCLINICALBERT = "emilyalsentzer/Bio_ClinicalBERT"


def build_report_classifier(num_labels: int, model_name: str = BIOCLINICALBERT):
    """Build a tokenizer and sequence classifier without loading patient text."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    return tokenizer, model
