import cv2
import numpy as np
import os
import json

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

    dataset_path = 'dataset'
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]

    faces = []
    labels = []
    label_map = {}
    label_reverse_map = {}
    label_count = 0

    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        real_id = image_path.split('.')[1]  # e.g. name.1.25.jpg

        if real_id not in label_map:
            label_map[real_id] = label_count
            label_reverse_map[label_count] = real_id
            label_count += 1

        labels.append(label_map[real_id])
        faces.append(img)

    recognizer.train(faces, np.array(labels))
    os.makedirs("trainer", exist_ok=True)
    recognizer.write("trainer/trainer.yml")

    with open("trainer/label_map.json", "w") as f:
        json.dump(label_reverse_map, f)

    print("Model trained.")

if __name__ == "__main__":
    train_model()
