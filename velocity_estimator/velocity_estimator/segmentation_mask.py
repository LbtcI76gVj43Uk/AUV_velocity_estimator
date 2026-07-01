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

# ---------------------------------------------------------------------------
# CityScapes trainIds (0-18, NICHT die labelIds 0-33!) für dynamische Klassen.
# Das Modell gibt trainIds zurück - eine kompaktere 19-Klassen-Kodierung.
# Quelle: cityscapesscripts/helpers/labels.py
# ---------------------------------------------------------------------------
DYNAMIC_TRAIN_IDS = {
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

_model = None
_processor = None
_MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"


def load_model(device: str = None):
    """Lädt das Modell einmalig (lazy loading, wird gecacht)."""
    global _model, _processor
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if _model is None:
        _processor = SegformerImageProcessor.from_pretrained(_MODEL_NAME)
        _model = SegformerForSemanticSegmentation.from_pretrained(_MODEL_NAME)
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
    logits = model(**inputs).logits  # Shape (1, 19, h', w') - NIEDRIGERE Auflösung!

    # SegFormer gibt die Logits in reduzierter Auflösung zurück -> zwingend
    # zurück auf Originalgröße hochskalieren, bevor man argmax nimmt:
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
    return np.isin(class_map, list(DYNAMIC_TRAIN_IDS.keys())).astype(np.uint8)


def mask_to_visual(mask: np.ndarray) -> np.ndarray:
    """Konvertiert die binäre Maske (0/1) in ein anzeigbares 0/255 uint8 Bild,
    in der Darstellung von Fig. 4b: Objekt = schwarz, Hintergrund = weiß."""
    visual = np.where(mask == 1, 0, 255).astype(np.uint8)
    return visual


def mask_to_visual(mask: np.ndarray) -> np.ndarray:
    """Konvertiert die binäre Maske (0/1) in ein anzeigbares 0/255 uint8 Bild,
    in der Darstellung von Fig. 4b: Objekt = schwarz, Hintergrund = weiß."""
    return np.where(mask == 1, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Schnelltest der Masken-LOGIK mit einer simulierten Modell-Ausgabe
# (kein echtes Modell/Internet nötig, um get_dynamic_mask() zu verifizieren).
# Der eigentliche Inferenz-Teil (predict_class_map) musst du bei dir lokal
# testen, da hier kein Internetzugriff zum Herunterladen der Gewichte besteht.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    h, w = 110, 220
    # Simulierte Modell-Ausgabe: größtenteils "road" (trainId 0),
    # mit einer "person" (11) und einem "car" (13) Region
    fake_class_map = np.zeros((h, w), dtype=np.uint8)
    fake_class_map[20:60, 40:60] = 11    # Person
    fake_class_map[50:90, 120:180] = 13  # Auto

    mask = get_dynamic_mask(fake_class_map)
    print("Maske Shape:", mask.shape, "dtype:", mask.dtype)
    print("Anteil dynamischer Pixel: %.1f%%" % (100 * mask.mean()))
    assert mask[30, 50] == 1, "Person sollte als dynamisch erkannt werden"
    assert mask[60, 150] == 1, "Auto sollte als dynamisch erkannt werden"
    assert mask[5, 5] == 0, "Straße sollte NICHT als dynamisch erkannt werden"
    print("✓ Masken-Logik-Tests bestanden.")

    import cv2
    visual = mask_to_visual(mask)
    cv2.imwrite("/home/claude/test_segmentation_mask.png", visual)
    print("Test-Visualisierung gespeichert unter test_segmentation_mask.png")
    print()
    print("Hinweis: predict_class_map() selbst (echte Modell-Inferenz) ist")
    print("hier nicht getestet, da kein Internetzugriff verfügbar ist, um")
    print("die SegFormer-Gewichte herunterzuladen. Bitte lokal testen mit:")
    print("  frame = cv2.cvtColor(cv2.imread('dein_frame.png'), cv2.COLOR_BGR2RGB)")
    print("  class_map = predict_class_map(frame)")
    print("  mask = get_dynamic_mask(class_map)")