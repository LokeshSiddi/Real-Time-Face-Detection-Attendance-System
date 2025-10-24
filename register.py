import cv2
import csv
import os


def register_student(student_id, student_name):
    face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)

    dataset_path = 'dataset'
    os.makedirs(dataset_path, exist_ok=True)

    sample_num = 0

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            sample_num += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{dataset_path}/{student_name}.{student_id}.{sample_num}.jpg", face_img)
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.imshow('Registering...', img)
        if cv2.waitKey(1) == 27 or sample_num >= 50:
            break

    # Save to student info CSV
    os.makedirs("data", exist_ok=True)
    with open("data/students.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([student_id, student_name])

    cam.release()
    cv2.destroyAllWindows()
    print("Faces captured.")

if __name__ == "__main__":
    student_id = input("Enter Student ID: ")
    student_name = input("Enter Student Name: ")
    register_student(student_id, student_name)
