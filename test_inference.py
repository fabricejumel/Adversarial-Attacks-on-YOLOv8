from ultralytics import YOLO
import cv2

# 1. Charger le modèle YOLOv8 nano pré-entraîné
print("Chargement du modèle YOLO...")
model = YOLO("yolov8n.pt")

# 2. Exécuter une inférence sur une image d'exemple (ici, une image par défaut d'Ultralytics : "bus.jpg")
print("Exécution de l'inférence sur l'image de test...")
results = model("https://ultralytics.com/images/bus.jpg")

# 3. Afficher les résultats textuels (boîtes, classes, confiances)
for r in results:
    boxes = r.boxes
    print(f"Nombre d'objets détectés : {len(boxes)}")
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]
        print(f" - Objet : {class_name} (Confiance : {conf:.2f})")

# 4. Sauvegarder l'image annotée
results[0].save(filename="resultat_inference.jpg")
print("Image annotée sauvegardée sous 'resultat_inference.jpg'.")
