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
from velocity_estimator.segmentation_mask import predict_class_map, get_dynamic_mask

def dilate_mask(mask_u8: np.ndarray, kernel_size: int = 15, iterations: int = 1) -> np.ndarray:
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
    return cv2.dilate(mask_u8, kernel, iterations=iterations)

def inpaint_flow(flow_rgb: np.ndarray, mask_u8: np.ndarray, inpaint_radius: int = 1) -> np.ndarray:
    """
    flow_rgb : (H, W, 3) RGB-kodierter Optical Flow, UNMASKIERT (Original)
    mask     : (H, W) binär, 1 = soll inpainted werden (dynamisches Objekt)
    inpaint_radius : Radius der Nachbarschaft, die für die Rekonstruktion
                      jedes Pixels berücksichtigt wird (Paper spezifiziert
                      keinen Wert - 3 ist OpenCVs Standardempfehlung)

    cv2.inpaint implementiert direkt Telea A. [11] mit flags=INPAINT_TELEA.
    """
    return cv2.inpaint(flow_rgb, mask_u8, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_NS)

def prepare_speed_estimator_input(
    frame_rgb: np.ndarray,
    flow_rgb: np.ndarray,
    dilation_kernel_size: int = 15,
    inpaint_radius: int = 1,
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
    # Vorab-Skalierung des Frames zur massiven Entlastung des SegFormers
    # Falls das Bild z.B. 1024x1024 ist, läuft SegFormer nun auf 220x110.
    small_frame = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    
    class_map = predict_class_map(small_frame)
    mask = get_dynamic_mask(class_map)

    flow_h, flow_w = flow_rgb.shape[:2]
    
    # Maske direkt auf die Zielauflösung des Flusses bringen
    if mask.shape != (flow_h, flow_w):
        mask = cv2.resize(mask, (flow_w, flow_h),
            interpolation=cv2.INTER_NEAREST,  # NEAREST! sonst keine reinen 0/1-Werte mehr
        )

    # Einmalige Multiplikation statt rechenintensiver Abfrage im Inpainter
    mask_u8 = mask * 255

    dilated = dilate_mask(mask_u8, kernel_size=dilation_kernel_size)
    inpainted = inpaint_flow(flow_rgb, dilated, inpaint_radius=inpaint_radius)

    if inpainted.shape[:2] != (target_size[1], target_size[0]):
        return cv2.resize(inpainted, target_size, interpolation=cv2.INTER_LINEAR)
    return inpainted

if __name__ == "__main__":
    pass
