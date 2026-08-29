"""1D CNN + LSTM ensemble for intraoperative hypotension prediction from
raw arterial waveforms, matching STEP-OP's real published architecture
(Lee et al., JMIR Medical Informatics 2021, PMC8517810, Seoul National
University -- built on VitalDB, the same dataset this project uses):
a weighted-average ensemble of a 1D CNN branch and an LSTM branch, both
operating on 20-second arterial-waveform windows (10,000 samples @
500Hz, confirmed directly from a real preprocessed shard), predicting
a hypotensive event 5 minutes ahead. STEP-OP's own reported ensemble
weights were 0.65 (LSTM) / 0.35 (CNN) -- this implementation makes that
weighting a learned parameter (a single scalar, sigmoid-constrained to
[0,1]) rather than fixing it, since there's no reason to assume this
project's data/preprocessing reproduces STEP-OP's exact split closely
enough for their specific weight to transfer, and a learned weight
costs nothing extra.

Each branch is intentionally lightweight relative to the CNN backbones
used elsewhere in this portfolio (EfficientNet-B3 for PE/trauma) --
this is a single 1D signal per exam, not a multi-hundred-slice image
sequence, so a much smaller 1D CNN + LSTM is the appropriate scale
here, not a reused 2D vision backbone.
"""
import torch
import torch.nn as nn


class CNNBranch(nn.Module):
    def __init__(self, in_channels: int = 1, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, hidden, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T) -> logits: (B,)"""
        feat = self.net(x).squeeze(-1)
        return self.head(feat).squeeze(-1)


class LSTMBranch(nn.Module):
    def __init__(self, hidden: int = 64, downsample: int = 10):
        super().__init__()
        # Raw 500Hz waveform (10,000 samples/window) is too long to feed an
        # LSTM one-sample-per-step -- downsample via a strided conv first
        # (standard practice for raw-waveform RNNs), then run the LSTM over
        # the resulting ~1000-step sequence.
        self.downsample = nn.Conv1d(1, 8, kernel_size=downsample, stride=downsample)
        self.lstm = nn.LSTM(input_size=8, hidden_size=hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T) -> logits: (B,)"""
        feat = self.downsample(x)  # (B, 8, T/downsample)
        feat = feat.transpose(1, 2)  # (B, T/downsample, 8)
        _, (h_n, _) = self.lstm(feat)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=-1)  # concat both directions' final hidden state
        return self.head(h_cat).squeeze(-1)


class STEPOPEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNBranch()
        self.lstm = LSTMBranch()
        # learned ensemble weight, sigmoid-constrained to [0,1] -- see module
        # docstring for why this isn't fixed at STEP-OP's reported 0.65/0.35
        self.raw_lstm_weight = nn.Parameter(torch.tensor(0.6))  # init near STEP-OP's reported value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T) raw waveform -> logits: (B,)"""
        cnn_logit = self.cnn(x)
        lstm_logit = self.lstm(x)
        w = torch.sigmoid(self.raw_lstm_weight)
        return w * lstm_logit + (1 - w) * cnn_logit
