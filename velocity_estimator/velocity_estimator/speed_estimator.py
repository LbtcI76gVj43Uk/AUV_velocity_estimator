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
from transformers import ViTModel

# Dataset und Training kommen aus kitti_dataset.py (nur dort importieren,
# wo tatsächlich trainiert wird - siehe kitti_dataset.KittiSpeedDataset/train)

from efficientnet_pytorch import EfficientNet


class EfficientNetFlowSpeed(nn.Module):
    """EfficientNet-Bx mit an 2 Eingabekanäle (roher Flow, kein Masking)
    angepasster erster Conv-Schicht und einem Ausgabe-Neuron (m/s)."""

    def __init__(self, variant: str = "b0"):
        super().__init__()
        self.variant = variant
        self.net = EfficientNet.from_pretrained(
            f"efficientnet-{variant}", in_channels=2, num_classes=1
        )

    def forward(self, x):
        return self.net(x)


def load_efficientnet_flow_speed(model_path: str, device: str = "cpu") -> "EfficientNetFlowSpeed":
    """Lädt einen mit efficientnet_baseline/train.py gespeicherten
    Checkpoint ({"state_dict", "variant"})."""
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        variant = checkpoint.get("variant", "b0")
        state_dict = checkpoint["state_dict"]
    else:
        variant = "b0"
        state_dict = checkpoint

    model = EfficientNetFlowSpeed(variant=variant)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Architektur 1: Bojarski et al. CNN (NVIDIA 2016)
# ---------------------------------------------------------------------------
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
        self._feature_size = self._get_feature_size()
        self.regressor = nn.Sequential(
            nn.Linear(self._feature_size, 100), nn.ELU(), nn.Dropout(dropout_p),
            nn.Linear(100, 50),                nn.ELU(), nn.Dropout(dropout_p),
            nn.Linear(50, 10),                 nn.ELU(),
            nn.Linear(10, 1),
        )

    def _get_feature_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 110, 220)
            return self.features(dummy).flatten(1).shape[1]

    def forward(self, x):
        return self.regressor(self.features(x).flatten(1))


def load_bojarski_cnn(model_path: str, device: str = "cpu") -> "BojarskiCNN":
    """Lädt einen mit kitti_dataset.train() gespeicherten Checkpoint.
    Akzeptiert sowohl das neue Format ({"state_dict", "dropout_p", ...})
    als auch einen rohen state_dict (ältere Checkpoints)."""
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        dropout_p = checkpoint.get("dropout_p", 0.0)
        state_dict = checkpoint["state_dict"]
    else:
        dropout_p = 0.0
        state_dict = checkpoint

    model = BojarskiCNN(dropout_p=dropout_p)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Architektur 2: Two-Stream (CNN + ViT), Fig. 6 im Paper
# ---------------------------------------------------------------------------
class TwoStreamNet(nn.Module):
    """
    Branch 1 (CNN):  RGB Flow -> Feature-Vektor (Motion-Details)
    Branch 2 (ViT):  Segmentierungsmaske -> Feature-Vektor (Regions of Interest)
    Fusion: Element-wise Multiplikation -> FC-Layer -> Geschwindigkeit
    """
    VIT_MODEL = "google/vit-base-patch16-224"
    VIT_HIDDEN = 768

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ELU(),
        )
        with torch.no_grad():
            cnn_out_size = self.cnn(torch.zeros(1, 3, 110, 220)).flatten(1).shape[1]

        self.vit = ViTModel.from_pretrained(self.VIT_MODEL)

        fusion_dim = 256
        self.cnn_proj = nn.Linear(cnn_out_size, fusion_dim)
        self.vit_proj = nn.Linear(self.VIT_HIDDEN, fusion_dim)
        self.regressor = nn.Sequential(
            nn.ELU(),
            nn.Linear(fusion_dim, 50), nn.ELU(),
            nn.Linear(50, 10),         nn.ELU(),
            nn.Linear(10, 1),
        )

    def forward(self, flow, mask):
        cnn_feat = self.cnn_proj(self.cnn(flow).flatten(1))
        mask_resized = torch.nn.functional.interpolate(
            mask.expand(-1, 3, -1, -1), size=(224, 224), mode="nearest"
        )
        vit_feat = self.vit_proj(
            self.vit(pixel_values=mask_resized).last_hidden_state[:, 0]
        )
        return self.regressor(cnn_feat * vit_feat)


def load_two_stream_net(model_path: str, device: str = "cpu") -> "TwoStreamNet":
    """Lädt einen mit kitti_dataset.train(..., mask_dir=...) gespeicherten
    TwoStreamNet-Checkpoint (Format: {"state_dict", "model_type", ...})."""
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    model = TwoStreamNet()
    model.load_state_dict(state_dict)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Shapes testen (ohne echte Daten)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== BojarskiCNN ===")
    cnn = BojarskiCNN()
    x = torch.rand(4, 3, 110, 220)
    out = cnn(x)
    assert out.shape == (4, 1)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Parameter: {sum(p.numel() for p in cnn.parameters()):,}")
    print("✓ BojarskiCNN Shape-Test bestanden\n")

    try:
        print("=== TwoStreamNet ===")
        two = TwoStreamNet()
        flow = torch.rand(2, 3, 110, 220)
        mask = torch.randint(0, 2, (2, 1, 110, 220)).float()
        out = two(flow, mask)
        assert out.shape == (2, 1)
        print(f"Output: {tuple(out.shape)}")
        print(f"Parameter: {sum(p.numel() for p in two.parameters()):,}")
        print("✓ TwoStreamNet Shape-Test bestanden")
    except Exception as e:
        print(f"TwoStreamNet-Test übersprungen (ViT-Download nötig): {e}")