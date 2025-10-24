import tkinter as tk
from tkinter import ttk
import csv
import os

def view_attendance():
    date = input("Enter date (YYYY-MM-DD): ")
    filename = f"attendance/Attendance_{date}.csv"

    if not os.path.exists(filename):
        print("No attendance for this date.")
        return

    root = tk.Tk()
    root.title(f"Attendance for {date}")

    tree = ttk.Treeview(root, columns=("ID", "Name", "Time"), show="headings")
    tree.heading("ID", text="Student ID")
    tree.heading("Name", text="Name")
    tree.heading("Time", text="Time")


    with open(filename, newline='') as f:
        reader = csv.reader(f)

        for row in reader:
            tree.insert("", tk.END, values=row)

    tree.pack()
    root.mainloop()

if __name__ == "__main__":
    view_attendance()
