import cv2
import numpy as np
import sys

# Vérifie les arguments
if len(sys.argv) < 2:
    print("Usage: python draw_mask.py image.jpg")
    sys.exit(1)

img_path = sys.argv[1]
image = cv2.imread(img_path)
if image is None:
    print("Erreur : image introuvable.")
    sys.exit(1)

drawing = False
radius = 10  # Taille du pinceau
mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Masque noir
display_img = image.copy()

def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(mask, (x, y), radius, 255, -1)               # sur le masque
        cv2.circle(display_img, (x, y), radius, (0, 0, 255), -1) # sur l'image
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow('Dessine la zone à supprimer')
cv2.setMouseCallback('Dessine la zone à supprimer', draw)

print("[🖱️] Clic gauche = dessiner | s = sauver | q = quitter")
print("[+]/- : changer la taille du pinceau (actuel =", radius, ")")

while True:
    cv2.imshow('Dessine la zone à supprimer', display_img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Sortie sans sauvegarde.")
        break
    elif key == ord('s'):
        mask_filename = "seam_carving_project/mask_" + img_path.split("/")[-1].split("\\")[-1].split('.')[0] + ".png"
        cv2.imwrite(mask_filename, mask)
        print(f"[✅] Masque sauvegardé sous '{mask_filename}'")
        break
    elif key == ord('+') or key == ord('='):  # = pour les claviers FR
        radius += 1
        print(f"[🖌️] Pinceau : {radius}px")
    elif key == ord('-'):
        radius = max(1, radius - 1)
        print(f"[🖌️] Pinceau : {radius}px")

cv2.destroyAllWindows()
