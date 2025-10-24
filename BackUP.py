import tkinter as tk
from tkinter import ttk, messagebox
import os
import cv2
import csv
import json
from datetime import datetime
import numpy as np

# ----- Constants -----
CASCADE_PATH = "haarcascade/haarcascade_frontalface_default.xml"
DATASET_PATH = "dataset"
TRAINER_PATH = "trainer"
STUDENT_FILE = "data/students.csv"
ATTENDANCE_DIR = "attendance"

# ----- Ensure directories -----
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(TRAINER_PATH, exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# ----- Global Variables -----
id_name_map = {}

# ----- Main App -----
class AttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition Attendance System")
        self.attributes('-fullscreen', True)
        self.bind("<Escape>", lambda e: self.destroy())

        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)
        self.frames = {}

        for F in (StartPage, RegisterPage, AttendancePage, ViewAttendancePage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, name):
        frame = self.frames[name]
        frame.tkraise()

# ----- Start Page -----
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="#f0f0f0")

        tk.Label(self, text="Face Recognition Attendance System", font=("Arial", 40, "bold")).pack(pady=60)

        btn_frame = tk.Frame(self, bg="#f0f0f0")
        btn_frame.pack(expand=True)

        top_buttons = tk.Frame(btn_frame, bg="#f0f0f0")
        top_buttons.pack()

        tk.Button(top_buttons, text="Register", font=("Arial", 20), width=20, height=2,
                  command=lambda: controller.show_frame("RegisterPage")).pack(side="left", padx=40)

        tk.Button(top_buttons, text="Take Attendance", font=("Arial", 20), width=20, height=2,
                  command=lambda: controller.show_frame("AttendancePage")).pack(side="right", padx=40)

        tk.Button(btn_frame, text="View Attendance", font=("Arial", 20), width=20, height=2,
                  command=lambda: controller.show_frame("ViewAttendancePage")).pack(pady=60)

# ----- Register Page -----
class RegisterPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # Left Input Panel
        input_frame = tk.Frame(self)
        input_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        tk.Label(input_frame, text="Student ID:", font=("Arial", 18)).pack(pady=10)
        self.id_entry = tk.Entry(input_frame, font=("Arial", 16))
        self.id_entry.pack()

        tk.Label(input_frame, text="Student Name:", font=("Arial", 18)).pack(pady=10)
        self.name_entry = tk.Entry(input_frame, font=("Arial", 16))
        self.name_entry.pack()

        tk.Button(input_frame, text="Start Capture", font=("Arial", 16),
                  command=self.capture_faces).pack(pady=20)

        # Right Camera Panel
        self.cam_label = tk.Label(self)
        self.cam_label.grid(row=0, column=1, sticky="nsew")

    def capture_faces(self):
        student_id = self.id_entry.get().strip()
        student_name = self.name_entry.get().strip()
        if not student_id or not student_name:
            messagebox.showerror("Error", "Please enter both ID and Name")
            return

        # Update student CSV
        rows = []
        exists = False
        if os.path.exists(STUDENT_FILE):
            with open(STUDENT_FILE, "r") as f:
                rows = [row for row in csv.reader(f) if row and row[0] != student_id]
                exists = any(row[0] == student_id for row in rows)

        rows.append([student_id, student_name])
        with open(STUDENT_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        # Remove old images
        for file in os.listdir(DATASET_PATH):
            if file.startswith(f"User.{student_id}."):
                os.remove(os.path.join(DATASET_PATH, file))

        cam = cv2.VideoCapture(0)
        face_detector = cv2.CascadeClassifier(CASCADE_PATH)
        count = 0

        while True:
            ret, img = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                count += 1
                cv2.imwrite(f"{DATASET_PATH}/User.{student_id}.{count}.jpg", gray[y:y+h, x:x+w])
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow("Registering...", img)
            if cv2.waitKey(1) == 27 or count >= 50:
                break

        cam.release()
        cv2.destroyAllWindows()

        train_model()
        messagebox.showinfo("Success", f"Captured and trained model for {student_name}")
        self.controller.show_frame("StartPage")

# ----- Train Model Automatically -----
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    label_map, rev_map = {}, {}
    label_id = 0

    for img in os.listdir(DATASET_PATH):
        path = os.path.join(DATASET_PATH, img)
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        id_ = img.split(".")[1]
        if id_ not in label_map:
            label_map[id_] = label_id
            rev_map[label_id] = id_
            label_id += 1
        faces.append(gray)
        labels.append(label_map[id_])

    recognizer.train(faces, np.array(labels))
    recognizer.write(f"{TRAINER_PATH}/trainer.yml")

    with open(f"{TRAINER_PATH}/label_map.json", "w") as f:
        json.dump(rev_map, f)

# ----- Attendance Page -----
class AttendancePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.label = tk.Label(self)
        self.label.pack(expand=True)

        self.after(500, self.recognize_faces)

    def recognize_faces(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(f"{TRAINER_PATH}/trainer.yml")
        with open(f"{TRAINER_PATH}/label_map.json", "r") as f:
            rev_map = json.load(f)

        names = {}
        if os.path.exists(STUDENT_FILE):
            with open(STUDENT_FILE, "r") as f:
                for row in csv.reader(f):
                    if row:
                        names[row[0]] = row[1]

        today = datetime.now().strftime("%Y-%m-%d")
        filename = f"{ATTENDANCE_DIR}/Attendance_{today}.csv"
        marked = set()
        if os.path.exists(filename):
            with open(filename, "r") as f:
                for row in csv.reader(f):
                    if row:
                        marked.add(row[0])

        def mark_attendance(sid):
            now = datetime.now().strftime("%H:%M:%S")
            name = names.get(sid, "Unknown")
            with open(filename, "a", newline="") as f:
                writer = csv.writer(f)
                if os.path.getsize(filename) == 0:
                    writer.writerow(["Student ID", "Name", "Time"])
                writer.writerow([sid, name, now])

        cam = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < 60:
            ret, img = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
                student_id = rev_map.get(str(id_))
                if student_id and student_id not in marked and conf < 70:
                    mark_attendance(student_id)
                    marked.add(student_id)
                    cv2.putText(img, f"Marked {student_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow("Taking Attendance...", img)
            if cv2.waitKey(1) == 27:
                break

        cam.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Done", "Attendance taken successfully")
        self.controller.show_frame("StartPage")

# ----- View Attendance Page -----
class ViewAttendancePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.controller = controller
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=4)
        self.grid_rowconfigure(0, weight=1)

        # Left: Dates list
        self.date_list = tk.Listbox(self, font=("Arial", 14))
        self.date_list.grid(row=0, column=0, sticky="nsew")
        self.date_list.bind("<<ListboxSelect>>", self.load_csv)

        # Right: Table view
        self.tree = ttk.Treeview(self, columns=("ID", "Name", "Time"), show="headings")
        self.tree.heading("ID", text="Student ID")
        self.tree.heading("Name", text="Name")
        self.tree.heading("Time", text="Time")
        self.tree.grid(row=0, column=1, sticky="nsew")

        self.load_dates()

    def load_dates(self):
        self.date_list.delete(0, tk.END)
        for file in sorted(os.listdir(ATTENDANCE_DIR)):
            if file.endswith(".csv"):
                self.date_list.insert(tk.END, file)

    def load_csv(self, event):
        selection = self.date_list.curselection()
        if not selection:
            return
        filename = self.date_list.get(selection[0])
        path = os.path.join(ATTENDANCE_DIR, filename)

        for row in self.tree.get_children():
            self.tree.delete(row)

        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 3:
                    self.tree.insert("", tk.END, values=(row[0], row[1], row[2]))

# ----- Run App -----
if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()
