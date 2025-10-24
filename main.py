import tkinter as tk
import subprocess
from tkinter import messagebox

# ---------- Functions for Buttons ----------

def run_register():
    try:
        subprocess.run(["python", "register.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to register student.\n{e}")

def run_train():
    try:
        subprocess.run(["python", "train.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model.\n{e}")

def run_recognize():
    try:
        subprocess.run(["python", "recognize.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to recognize faces.\n{e}")

def run_view():
    try:
        subprocess.run(["python", "view_attendance.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to view attendance.\n{e}")

def exit_app():
    if messagebox.askokcancel("Exit", "Do you really want to exit?"):
        root.destroy()

# ---------- Main Window Setup ----------

root = tk.Tk()
root.title("🎓 Face Recognition Attendance System")
root.geometry("400x350")
root.resizable(False, False)

# Optional: Set background color
root.configure(bg="#f0f0f0")

# ---------- Title Label ----------
title = tk.Label(root, text="Face Recognition Attendance", font=("Helvetica", 16, "bold"), pady=20, bg="#f0f0f0")
title.pack()

# ---------- Buttons ----------
btn_width = 30
btn_pad = 10

tk.Button(root, text="➕ Register Student", width=btn_width, command=run_register).pack(pady=btn_pad)
tk.Button(root, text="🧠 Train Model", width=btn_width, command=run_train).pack(pady=btn_pad)
tk.Button(root, text="🎥 Take Attendance", width=btn_width, command=run_recognize).pack(pady=btn_pad)
tk.Button(root, text="📊 View Attendance", width=btn_width, command=run_view).pack(pady=btn_pad)
tk.Button(root, text="❌ Exit", width=btn_width, fg="white", bg="red", command=exit_app).pack(pady=btn_pad+5)

# ---------- Run the App ----------
root.mainloop()
