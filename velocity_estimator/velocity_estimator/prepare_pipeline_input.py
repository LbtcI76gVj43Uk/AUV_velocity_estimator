"""
Dilatation + Inpainting (Schritt 4+5) -> fertiger Input für Speed Estimator
=============================================================================
Implementiert die letzten beiden Vorverarbeitungsschritte aus dem Paper und
kombiniert ALLES (Segmentierung -> Maskierung -> Dilatation -> Inpainting)
zu einer einzigen Funktion, deren Ausgabe direkt in den Speed-Estimator-CNN
gegeben werden kann (Methode 5 in Tabelle II, bestes Ergebnis: MAE 0.845).

Wichtig zum Verständnis: Inpainting braucht NICHT den bereits geschwärzten
Flow aus Schritt 3 als Input. cv2.inpaint() bekommt den ORIGINALEN
(unmaskierten) Flow plus die Maske, und füllt selbst die maskierten Pixel
mit plausiblen Werten aus der Umgebung. Schritt 3 (mask_optical_flow.py)
war nur zur Veranschaulichung/zum Testen gedacht (Fig. 4c im Paper) - für
die finale Pipeline wird er nicht gebraucht.
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Schritt 4: Dilatation der Maske (Gleichung 3 im Paper)
# ---------------------------------------------------------------------------
def dilate_mask(mask: np.ndarray, kernel_size: int = 15, iterations: int = 1) -> np.ndarray:
    """
    Vergrößert die dynamischen Objekt-Regionen in der Maske, um Flow-Reste
    am Rand bewegter Objekte mit zu erfassen (im Paper: Fig. 4c vs Fig. 5c -
    in 4c sieht man noch Flow-Reste um die maskierten Personen, in 5c nicht
    mehr).

    Hinweis: kernel_size ist im Paper NICHT spezifiziert - das ist ein
    Hyperparameter, den du auf deinen eigenen Validierungsdaten testen
    solltest. 15 ist ein vernünftiger Startwert für 220x110px Bilder;
    bei höherer Auflösung proportional größer wählen.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)


# ---------------------------------------------------------------------------
# Schritt 5: Inpainting (Telea's Fast Marching Method, Gleichung 1+2)
# ---------------------------------------------------------------------------
def inpaint_flow(flow_rgb: np.ndarray, mask: np.ndarray, inpaint_radius: int = 3) -> np.ndarray:
    """
    flow_rgb : (H, W, 3) RGB-kodierter Optical Flow, UNMASKIERT (Original)
    mask     : (H, W) binär, 1 = soll inpainted werden (dynamisches Objekt)
    inpaint_radius : Radius der Nachbarschaft, die für die Rekonstruktion
                      jedes Pixels berücksichtigt wird (Paper spezifiziert
                      keinen Wert - 3 ist OpenCVs Standardempfehlung)

    cv2.inpaint implementiert direkt Telea A. [11] mit flags=INPAINT_TELEA.
    """
    mask_u8 = (mask * 255).astype(np.uint8)
    return cv2.inpaint(flow_rgb, mask_u8, inpaintRadius=inpaint_radius,
                        flags=cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# Komplette Pipeline: Frame + Flow -> fertiger Speed-Estimator-Input
# ---------------------------------------------------------------------------
def prepare_speed_estimator_input(
    frame_rgb: np.ndarray,
    flow_rgb: np.ndarray,
    dilation_kernel_size: int = 15,
    inpaint_radius: int = 3,
    target_size: tuple = (220, 110),  # (W, H), wie im Paper
):
    """
    Führt die komplette Vorverarbeitung durch: Segmentierung -> Dilatation
    -> Inpainting. Importiert die Funktionen aus den vorherigen Skripten.

    frame_rgb : (H, W, 3) das erste der beiden konsekutiven Frames
                (selbe Quelle wie für die Flow-Berechnung benutzt!)
    flow_rgb  : (H', W', 3) RGB-kodierter Optical Flow zwischen frame_rgb
                und dem darauffolgenden Frame

    frame_rgb und flow_rgb dürfen UNTERSCHIEDLICHE Auflösungen haben (z.B.
    wenn der Flow schon vorab herunterskaliert wurde) - die Maske wird
    automatisch auf die Auflösung von flow_rgb angepasst, bevor inpainting
    läuft, da cv2.inpaint identische Größen von Bild und Maske verlangt.

    Rückgabe  : (target_size[1], target_size[0], 3) uint8 Array,
                bereit für den CNN Speed Estimator
    """
    from segmentation_mask import predict_class_map, get_dynamic_mask

    class_map = predict_class_map(frame_rgb)
    mask = get_dynamic_mask(class_map)

    # WICHTIG: Maske auf die Auflösung von flow_rgb bringen, nicht auf die
    # von frame_rgb - beide können unterschiedlich groß sein.
    flow_h, flow_w = flow_rgb.shape[:2]
    if mask.shape != (flow_h, flow_w):
        mask = cv2.resize(
            mask.astype(np.uint8), (flow_w, flow_h),
            interpolation=cv2.INTER_NEAREST,  # NEAREST! sonst keine reinen 0/1-Werte mehr
        )

    dilated = dilate_mask(mask, kernel_size=dilation_kernel_size)
    inpainted = inpaint_flow(flow_rgb, dilated, inpaint_radius=inpaint_radius)

    # Auf die im Paper genutzte Zielauflösung bringen
    resized = cv2.resize(inpainted, target_size, interpolation=cv2.INTER_LINEAR)
    return resized


# ---------------------------------------------------------------------------
# Test mit synthetischen Daten (gleiches Setup wie in mask_optical_flow.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    h, w = 110, 220

    yy, xx = np.mgrid[0:h, 0:w]
    fake_flow = np.zeros((h, w, 3), dtype=np.uint8)
    fake_flow[..., 0] = (xx / w * 255).astype(np.uint8)
    fake_flow[..., 1] = (yy / h * 255).astype(np.uint8)
    fake_flow[..., 2] = 180
    fake_flow[50:90, 120:180] = [255, 0, 255]  # simuliertes Auto

    fake_mask = np.zeros((h, w), dtype=np.uint8)
    fake_mask[50:90, 120:180] = 1

    # --- Dilatation testen ---
    dilated = dilate_mask(fake_mask, kernel_size=15)
    assert dilated.sum() > fake_mask.sum(), "Dilatation sollte die Maske vergrößern"
    print(f"✓ Dilatation: Maske von {fake_mask.sum()} auf {dilated.sum()} Pixel vergrößert")

    # --- Inpainting testen ---
    inpainted = inpaint_flow(fake_flow, dilated, inpaint_radius=3)
    assert inpainted.shape == fake_flow.shape
    # Der vormals magenta-farbene Bereich sollte jetzt NICHT mehr magenta sein,
    # sondern an die umliegenden Gradientenfarben angeglichen
    center_pixel_before = fake_flow[70, 150]
    center_pixel_after = inpainted[70, 150]
    assert not np.array_equal(center_pixel_before, center_pixel_after), (
        "Inpainting sollte den Objektbereich verändert haben"
    )
    assert not np.array_equal(center_pixel_after, [0, 0, 0]), (
        "Inpainting sollte NICHT einfach schwarz sein (das wäre nur Maskierung)"
    )
    print(f"✓ Inpainting: Pixel (70,150) war {center_pixel_before} (Magenta-Objekt), "
          f"ist jetzt {center_pixel_after} (an Umgebung angeglichen)")

    # --- Visueller Vergleich: Original | Dilatierte Maske | Maskiert | Inpainted ---
    mask_vis = np.stack([dilated * 255] * 3, axis=-1).astype(np.uint8)
    masked_vis = fake_flow.copy()
    masked_vis[dilated == 1] = 0
    comparison = np.concatenate([fake_flow, mask_vis, masked_vis, inpainted], axis=1)
    cv2.imwrite("/home/claude/full_pipeline_comparison.png",
                cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print("Vergleichsbild gespeichert: full_pipeline_comparison.png")
    print("(Original Flow | Dilatierte Maske | Maskiert (schwarz) | Inpainted (final))")