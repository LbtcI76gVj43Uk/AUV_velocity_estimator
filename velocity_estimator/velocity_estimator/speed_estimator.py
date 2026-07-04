"""
Speed Estimator: Bojarski-CNN + Two-Stream-Architektur (Abschnitt V.C)
========================================================================
Implementiert beide Architekturen aus dem Paper:
  - BojarskiCNN: 5 Conv-Layer + 3 FC-Layer + 1 Output-Neuron (MAE-Loss)
  - TwoStreamNet: CNN-Branch für Flow + ViT-Branch für Maske,
                  Feature-Vektoren werden multipliziert, dann FC-Layer

EMPFEHLUNG:
  < ~5.000 Samples  -> BojarskiCNN (weniger Parameter, weniger Overfitting)
  > ~5.000 Samples  -> TwoStreamNet (ViT braucht mehr Daten zum Generalisieren)

Dataset + Training: siehe kitti_dataset.py (KittiSpeedDataset, train())

Input (beide Architekturen):
  BojarskiCNN  : (B, 3, 110, 220) - preprocessed RGB Flow
  TwoStreamNet : (B, 3, 110, 220) + (B, 1, 110, 220) - Flow + binäre Maske

Output: (B, 1) - geschätzte Geschwindigkeit in m/s
"""

import torch
import torch.nn as nn

class BojarskiCNN(nn.Module):
    """
    5 Convolutional Layer + 3 Fully Connected Layer + 1 Output-Neuron.
    Input-Shape: (B, 3, 110, 220) - wie im Paper (Tabelle I/II).

    dropout_p : Dropout-Wahrscheinlichkeit in den FC-Layern.
                0.0 = kein Dropout (Original-Paper), 0.3-0.5 empfohlen
                wenn Train-MAE << Val-MAE (Overfitting).
                Dropout ist nur während model.train() aktiv,
                bei model.eval() (Inferenz/Validation) automatisch deaktiviert.
    """

    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        self.dropout_p = dropout_p  # als Attribut speichern für Checkpoint
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ELU(),
        )
        self._feature_size = 8960 # 8960 ist die exakte Ausgabegröße für den (110, 220) Input
        self.regressor = nn.Sequential(
            nn.Linear(self._feature_size, 100), nn.ELU(), nn.Dropout(dropout_p),
            nn.Linear(100, 50), nn.ELU(), nn.Dropout(dropout_p),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.regressor(self.features(x).flatten(1))
