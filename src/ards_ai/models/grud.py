"""Compact GRU-D implementation for optional temporal modeling."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install the temporal extra: pip install -e '.[temporal]'") from exc


class GRUD(nn.Module):
    """GRU-D style recurrent model with learned input and hidden-state decay."""

    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_decay = nn.Linear(input_size, input_size)
        self.hidden_decay = nn.Linear(input_size, hidden_size)
        self.cell = nn.GRUCell(input_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        deltas: torch.Tensor,
        feature_means: torch.Tensor,
    ) -> torch.Tensor:
        """Return one logit per sequence.

        Inputs have shape [batch, time, feature], except feature_means [feature].
        """
        batch_size, time_steps, _ = values.shape
        hidden = values.new_zeros(batch_size, self.hidden_size)
        last_observed = feature_means.expand(batch_size, -1)

        for step in range(time_steps):
            value_t = values[:, step]
            mask_t = masks[:, step]
            delta_t = deltas[:, step]

            gamma_x = torch.exp(-torch.relu(self.input_decay(delta_t)))
            gamma_h = torch.exp(-torch.relu(self.hidden_decay(delta_t)))
            hidden = gamma_h * hidden

            decayed = gamma_x * last_observed + (1.0 - gamma_x) * feature_means
            imputed = mask_t * value_t + (1.0 - mask_t) * decayed
            last_observed = mask_t * value_t + (1.0 - mask_t) * last_observed

            hidden = self.cell(torch.cat([imputed, mask_t], dim=-1), hidden)

        return self.output(self.dropout(hidden)).squeeze(-1)
