import cv2
import os
import numpy as np
from datetime import datetime
import time
import csv
import json

# Load label map
with open("trainer/label_map.json", "r") as f:
    label_map = json.load(f)  # label -> actual_id

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

# Attendance file for today
today = datetime.now().strftime("%Y-%m-%d")
filename = f"attendance/Attendance_{today}.csv"
already_marked = set()

# Load existing attendance
if os.path.exists(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                already_marked.add(row[0])  # First column = ID


# Load student ID → Name mapping
id_name_map = {}
try:
    with open("data/students.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                id_name_map[row[0]] = row[1]
except FileNotFoundError:
    print("[WARNING] students.csv not found.")


# Attendance writer
def mark_attendance(student_id):
    now = datetime.now()
    time_str = now.strftime("%H:%M:%S")
    student_name = id_name_map.get(student_id, "Unknown")
    os.makedirs("attendance", exist_ok=True)
    write_header = not os.path.exists(filename)


    try:
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Student ID", "Name", "Time"])
            writer.writerow([student_id, student_name, time_str])
        print(f"[INFO] Marked attendance for {student_id} at {time_str}")
    except PermissionError:
        print(f"[ERROR] Cannot write to {filename}. Please close it if it's open.")


recognized = False
start_time = time.time()

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        student_id = label_map.get(str(id_), "Unknown")

        if confidence < 60 and student_id != "Unknown":
            if student_id not in already_marked:
                mark_attendance(student_id)
                already_marked.add(student_id)
                recognized = True

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"ID: {student_id}", (x+5, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            break
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(frame, "Unknown", (x+5, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Attendance System", frame)

    if recognized:
        cv2.waitKey(1000)  # Wait a second to show the recognized face
        break

    if cv2.waitKey(1) == 27:  # ESC to exit manually
        break

    if time.time() - start_time > 5:  # Run for 60 seconds
        print("TimeOut: Stopping recognition.")
        break

cam.release()
cv2.destroyAllWindows()





# import cv2
# import numpy as np
# import os
# import csv
# import time
# from datetime import datetime
#
# def mark_attendance(student_id):
#     now = datetime.now()
#     dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
#     date = now.strftime('%Y-%m-%d')
#
#     already_marked = set()
#
#     filename = f"attendance/Attendance_{date}.csv"
#     os.makedirs("attendance", exist_ok=True)
#
#     if not os.path.exists(filename):
#         with open(filename, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(["Student ID", "Time"])
#
#     with open(filename, 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([student_id, dt_string])
#
# def recognize():
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read("trainer/trainer.yml")
#     face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
#
#     cam = cv2.VideoCapture(0)
#
#     start_time = time.time()
#
#     while True:
#         ret, img = cam.read()
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.2, 5)
#
#         for (x,y,w,h) in faces:
#             id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
#             if confidence < 60:
#                 mark_attendance(id)
#                 label = f"ID: {id}"
#             else:
#                 label = "Unknown"
#
#             cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
#             cv2.putText(img, label, (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#
#         cv2.imshow('Recognizing...', img)
#         if cv2.waitKey(1) == 27:
#             break
#
#         if time.time() - start_time > 30:  # Run for 60 seconds
#             print("TimeOut: Stopping recognition.")
#             break
#
#     cam.release()
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     recognize()
