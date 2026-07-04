"""
Semantic Segmentation Mask Extraction (Schritt 2 der Pipeline)
================================================================
GENERALISIERTE Version: erzeugt die binäre Maske der "dynamischen" Objekte
(Personen, Fahrzeuge) per Modell-INFERENZ auf dem rohen RGB-Frame -
funktioniert daher für JEDE Quelle (KITTI, CityScapes, eigenes
Dashcam-Footage), nicht nur für Datensätze mit mitgelieferten Labels.

Modell: SegFormer (b0), vortrainiert auf CityScapes (19 Klassen, "trainIds").
Diese Klassen entsprechen genau dem, was im Paper als "dynamische" Objekte
(other vehicles and humans, Abschnitt V.B) maskiert wird. Da das Modell auf
urbanen Fahrszenen aus Fahrzeugperspektive trainiert ist, generalisiert es
gut auf KITTI & andere Dashcam-Daten (ähnliche Domäne).

Benötigt: pip install transformers --break-system-packages
Lädt beim ersten Aufruf automatisch die Gewichte von HuggingFace (Internet
nötig, danach lokal gecacht, ~14 MB für b0).
"""

import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# Als NumPy-Array für schnelleren C-Level-Abgleich in np.isin
DYNAMIC_TRAIN_IDS = np.array([11, 12, 13, 14, 15, 16, 17, 18], dtype=np.uint8)

_model = None
_processor = None
_MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

def load_model(device: str = None):
    global _model, _processor
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if _model is None:
        _processor = SegformerImageProcessor.from_pretrained(_MODEL_NAME)
        _model = SegformerForSemanticSegmentation.from_pretrained(_MODEL_NAME)
        # torch.inference_mode() kompatibel vorbereiten
        _model = _model.to(device).eval()
    return _model, _processor

@torch.no_grad()
def predict_class_map(frame_rgb: np.ndarray, device: str = None) -> np.ndarray:
    """
    Sagt für ein beliebiges RGB-Frame (egal aus welchem Datensatz) die
    Klasse pro Pixel vorher.

    frame_rgb : np.ndarray, Shape (H, W, 3), RGB, uint8
    Rückgabe  : np.ndarray, Shape (H, W), CityScapes trainIds (0-18)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(device)

    inputs = processor(images=frame_rgb, return_tensors="pt").to(device)
    
    # Inferenz beschleunigen
    outputs = model(**inputs)
    logits = outputs.logits 

    # Direkt auf der GPU argmax ausführen, falls CUDA aktiv ist (spart CPU-Kopierzeit)
    upsampled = torch.nn.functional.interpolate(
        logits, size=frame_rgb.shape[:2], mode="bilinear", align_corners=False
    )
    class_map = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return class_map

def get_dynamic_mask(class_map: np.ndarray) -> np.ndarray:
    """
    Erzeugt aus der Klassen-Map (egal ob von Ground-Truth oder Modell-
    Inferenz) eine binäre Maske der dynamischen Objekte.

    Rückgabe-Konvention: 1 = dynamisches Objekt, 0 = statischer Hintergrund.
    Diese Konvention lässt sich direkt mit cv2.dilate() und zum Maskieren
    des Optical Flow weiterverwenden.
    """
    return np.isin(class_map, DYNAMIC_TRAIN_IDS).astype(np.uint8)

if __name__ == "__main__":
    pass
