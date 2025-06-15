import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from datetime import datetime
import numpy as np
import os
from keras.models import load_model
import cv2
import json
import os.path

# Load class labels
with open("labels.txt", "r") as f:
    class_names = [line.strip().split(' ', 1)[-1].strip() for line in f]   # 1. Remove leading/trailing whitespace from each line
                                                                            # 2. Split the line at the first space; take the part after the space (e.g., skip label number)

# Database functions
def load_student_database():
    if os.path.exists("student_database.json"):
        try:
            with open("student_database.json", "r") as f:
                return json.load(f)
        except Exception as e:
            print("❌ Error loading student_database.json:", str(e))
            return None  # Or raise an error instead
    else:
        print("❌ student_database.json not found.")
        return None  # Or raise FileNotFoundError

def save_student_database(db):
    with open("student_database.json", "w") as f:
        json.dump(db, f, indent=2)

# Initialize database and model
student_database = load_student_database()
attendance_records = []
rules = [
    {"rule": "3 unexcused absences = warning", "threshold": 3, "consequence": "Warning"},
    {"rule": "5 unexcused absences = meeting", "threshold": 5, "consequence": "Meeting with supervisor"},
    {"rule": "7 unexcused absences = disciplinary action", "threshold": 7, "consequence": "Disciplinary action"}
]
model = load_model("keras_model.h5", compile=False) if os.path.exists("keras_model.h5") else None

class IAESApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Intelligent Attendance Evaluation System (IAES)")
        self.geometry("1200x800")
        self.configure(bg="#e6f2ff")
        self.frames = {}

        for F in (WelcomePage, AttendancePage):
            frame = F(self)
            self.frames[F] = frame
            frame.place(relwidth=1, relheight=1)

        self.show_frame(WelcomePage)

    def show_frame(self, page):
        frame = self.frames[page]
        frame.tkraise()

class WelcomePage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="#e6f2ff")
        label = tk.Label(self, text="Welcome to IAES", font=("Arial", 28, "bold"), bg="#e6f2ff", fg="#003366")
        label.pack(pady=50)

        description = tk.Label(
            self,
            text="Evaluate and monitor attendance intelligently with expert rules.",
            font=("Arial", 16),
            bg="#e6f2ff",
            fg="#555555",
            justify="center"
        )
        description.pack(pady=30)

        proceed_button = tk.Button(
            self,
            text="Start Attendance Evaluation",
            font=("Arial", 16, "bold"),
            bg="#003366",
            fg="#ffffff",
            command=lambda: parent.show_frame(AttendancePage),
            width=30,
        )
        proceed_button.pack(pady=50)

class AttendancePage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="#e6f2ff")
        title = tk.Label(self, text="IAES - Attendance Panel", font=("Arial", 24, "bold"), bg="#e6f2ff", fg="#003366")
        title.pack(pady=10)

        # Create tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)
        
        style = ttk.Style()
        style.configure("TNotebook", background="#e6f2ff")
        style.configure("TNotebook.Tab", background="#cce5ff", padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", "#99ccff")])
        
        # Tabs setup
        self.face_recognition_tab = ttk.Frame(self.notebook, style="TFrame")
        self.manual_entry_tab = ttk.Frame(self.notebook, style="TFrame")
        self.manage_db_tab = ttk.Frame(self.notebook, style="TFrame")
        
        self.notebook.add(self.face_recognition_tab, text="Face Recognition")
        self.notebook.add(self.manual_entry_tab, text="Manual Entry")
        self.notebook.add(self.manage_db_tab, text="Manage Student detail")
        
        self.setup_face_recognition_tab()
        self.setup_manual_entry_tab()
        self.setup_manage_db_tab()
        
        # Attendance Display
        display_frame = tk.Frame(self, bg="#e6f2ff")
        display_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        tk.Label(display_frame, text="Attendance Records:", font=("Arial", 14, "bold"), 
                bg="#e6f2ff", fg="#003366").pack(anchor="w")
        
        self.attendance_display = tk.Text(display_frame, height=15, width=120, font=("Arial", 12))
        self.attendance_display.pack(pady=5)
        
        # Action Buttons
        button_frame = tk.Frame(self, bg="#e6f2ff")
        button_frame.pack(pady=10)
        


    def setup_face_recognition_tab(self):
        tab = self.face_recognition_tab
        tab.configure(style="TFrame")
        
        # Image display frame
        image_display_frame = tk.Frame(tab, bg="#e6f2ff", width=400, height=300)
        image_display_frame.pack(pady=10)
        image_display_frame.pack_propagate(False)
        
        self.image_label = tk.Label(image_display_frame, bg="#e6f2ff", relief="solid", bd=1)
        self.image_label.pack(fill="both", expand=True)
        
        # Button frame
        btn_frame = tk.Frame(tab)
        btn_frame.pack(fill="x", padx=20, pady=10)
        
        btn_capture = tk.Button(
            btn_frame,
            text="Capture Image",
            font=("Arial", 12),
            bg="#2196F3",
            fg="#ffffff",
            command=self.capture_image,
            width=15
        )
        btn_capture.pack(side="left", padx=10)
        
        btn_select = tk.Button(
            btn_frame,
            text="Select Image",
            font=("Arial", 12),
            bg="#2196F3",
            fg="#ffffff",
            command=self.select_and_classify,
            width=15
        )
        btn_select.pack(side="left", padx=10)
        
        # Detection results
        self.result_label = tk.Label(btn_frame, text="", font=("Arial", 12), fg="#003366")
        self.result_label.pack(side="left", padx=20, fill="x", expand=True)
        
        # Action buttons
        action_frame = tk.Frame(btn_frame)
        action_frame.pack(side="right")
        
        btn_present = tk.Button(
            action_frame,
            text="Mark Present",
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="#ffffff",
            command=lambda: self.mark_attendance("Present"),
            width=15
        )
        btn_present.pack(side="left", padx=5)
        
        btn_absent = tk.Button(
            action_frame,
            text="Mark Absent",
            font=("Arial", 12, "bold"),
            bg="#f44336",
            fg="#ffffff",
            command=lambda: self.mark_attendance("Absent"),
            width=15
        )
        btn_absent.pack(side="left", padx=5)

    def setup_manual_entry_tab(self):
        tab = self.manual_entry_tab
        form_frame = tk.Frame(tab)
        form_frame.pack(pady=20)
        
        # Student selection
        tk.Label(form_frame, text="Select Student:", font=("Arial", 14), fg="#003366").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.student_var = tk.StringVar()
        
        # Create combobox first
        self.combo_student = ttk.Combobox(form_frame, textvariable=self.student_var, font=("Arial", 14), width=30)
        self.combo_student.grid(row=0, column=1, padx=10, pady=5)
        
        # Then update its values
        self.update_manual_entry_combobox()
        self.combo_student.bind("<<ComboboxSelected>>", self.on_student_select)
        
        # Student ID
        tk.Label(form_frame, text="Student ID:", font=("Arial", 14), fg="#003366").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.entry_id = tk.Entry(form_frame, font=("Arial", 14), width=32, state="normal", bg="#ffffff")
        self.entry_id.grid(row=1, column=1, padx=10, pady=5)
        
        # Absence count display
        tk.Label(form_frame, text="Total Absences:", font=("Arial", 14), fg="#003366").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.absence_count = tk.Label(form_frame, text="0", font=("Arial", 14),  fg="#003366")
        self.absence_count.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        
        # Status
        tk.Label(form_frame, text="Status:", font=("Arial", 14), fg="#003366").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.combo_status = ttk.Combobox(form_frame, font=("Arial", 14), width=28)
        self.combo_status["values"] = ("Present", "Absent")
        self.combo_status.grid(row=3, column=1, padx=10, pady=5)
        
        # Manual entry button
        btn_frame = tk.Frame(tab)
        btn_frame.pack(pady=10)
        
        btn_mark = tk.Button(
            btn_frame,
            text="Mark Attendance",
            font=("Arial", 14, "bold"),
            bg="#003366",
            fg="#ffffff",
            command=self.mark_manual_attendance,
            width=20
        )
        btn_mark.pack()

    def setup_manage_db_tab(self):
        tab = self.manage_db_tab
        tab.configure(style="TFrame")
        
        # Form for adding new students
        form_frame = tk.Frame(tab)
        form_frame.pack(pady=20, padx=20, fill="x")
        
        tk.Label(form_frame, text="Add New Student:", font=("Arial", 14, "bold"), 
                 fg="#003366").grid(row=0, column=0, columnspan=2, pady=10, sticky="w")
        
        # Name field
        tk.Label(form_frame, text="Full Name:", font=("Arial", 12), fg="#003366").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.new_name_var = tk.StringVar()
        entry_name = tk.Entry(form_frame, textvariable=self.new_name_var, font=("Arial", 12), width=30)
        entry_name.grid(row=1, column=1, padx=5, pady=5)
        
        # ID field
        tk.Label(form_frame, text="Student ID:", font=("Arial", 12), fg="#003366").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.new_id_var = tk.StringVar()
        entry_id = tk.Entry(form_frame, textvariable=self.new_id_var, font=("Arial", 12), width=30)
        entry_id.grid(row=2, column=1, padx=5, pady=5)
        
        # Initial absences
        tk.Label(form_frame, text="Initial Absences:", font=("Arial", 12), fg="#003366").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.new_absences_var = tk.StringVar(value="0")
        entry_absences = tk.Entry(form_frame, textvariable=self.new_absences_var, font=("Arial", 12), width=30)
        entry_absences.grid(row=3, column=1, padx=5, pady=5)
        
        # Add button
        btn_add = tk.Button(
            form_frame,
            text="Add Student",
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="#ffffff",
            command=self.add_new_student,
            width=15
        )
        btn_add.grid(row=4, column=1, padx=5, pady=15, sticky="w")
        
        # Database display
        db_display_frame = tk.Frame(tab)
        db_display_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Treeview to display student database
        columns = ("Name", "ID", "Absences")
        self.db_tree = ttk.Treeview(db_display_frame, columns=columns, show="headings")
        
        # Define headings
        self.db_tree.heading("Name", text="Student Name")
        self.db_tree.heading("ID", text="Student ID")
        self.db_tree.heading("Absences", text="Absences")
        
        # Set column widths
        self.db_tree.column("Name", width=200, anchor="center")
        self.db_tree.column("ID", width=150, anchor="center")
        self.db_tree.column("Absences", width=100, anchor="center")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(db_display_frame, orient="vertical", command=self.db_tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.db_tree.configure(yscrollcommand=scrollbar.set)
        
        self.db_tree.pack(fill="both", expand=True)
        
        # Populate the treeview
        self.populate_db_tree()
        
        # Delete button
        btn_frame = tk.Frame(db_display_frame)
        btn_frame.pack(pady=10)
        
        btn_delete = tk.Button(
            btn_frame,
            text="Delete Selected",
            font=("Arial", 12),
            bg="#f44336",
            fg="#ffffff",
            command=self.delete_selected_student,
            width=15
        )
        btn_delete.pack()

    def populate_db_tree(self):
        # Clear existing items
        for item in self.db_tree.get_children():
            self.db_tree.delete(item)
        
        # Add students from database
        for name, info in student_database.items():
            self.db_tree.insert("", "end", values=(name, info["id"], info["absences"]))

    def add_new_student(self):
        name = self.new_name_var.get().strip()
        student_id = self.new_id_var.get().strip()
        absences = self.new_absences_var.get().strip()
        
        # Validate input
        if not name:
            messagebox.showerror("Error", "Please enter a name")
            return
        if not student_id:
            messagebox.showerror("Error", "Please enter a student ID")
            return
        if not absences.isdigit():
            messagebox.showerror("Error", "Absences must be a number")
            return
            
        # Check if name or ID already exists
        if name in student_database:
            messagebox.showerror("Error", "Student name already exists in database")
            return
            
        for student_info in student_database.values():
            if student_info["id"] == student_id:
                messagebox.showerror("Error", "Student ID already exists in database")
                return
        
        # Add to database
        student_database[name] = {
            "id": student_id,
            "absences": int(absences)
        }
        
        # Save to file
        save_student_database(student_database)
        
        # Update UI
        self.populate_db_tree()
        self.update_manual_entry_combobox()
        
        # Clear form
        self.new_name_var.set("")
        self.new_id_var.set("")
        self.new_absences_var.set("0")
        
        messagebox.showinfo("Success", f"Added new student: {name}")

    def delete_selected_student(self):
        selected = self.db_tree.selection()
        if not selected:
            messagebox.showerror("Error", "Please select a student to delete")
            return
            
        # Confirm deletion
        item = self.db_tree.item(selected[0])
        name = item["values"][0]
        
        if messagebox.askyesno("Confirm", f"Delete student {name}? This cannot be undone."):
            # Remove from database
            del student_database[name]
            
            # Save to file
            save_student_database(student_database)
            
            # Update UI
            self.populate_db_tree()
            self.update_manual_entry_combobox()
            
            messagebox.showinfo("Success", f"Deleted student: {name}")

    def update_manual_entry_combobox(self):
        student_names = list(student_database.keys())
        student_names.sort()
        self.combo_student["values"] = student_names

    def on_student_select(self, event):
        student_name = self.student_var.get()
        student_info = student_database.get(student_name, {})
        student_id = student_info.get("id", "")
        absences = student_info.get("absences", 0)
        
        self.entry_id.config(state="normal")
        self.entry_id.delete(0, tk.END)
        self.entry_id.insert(0, student_id)
        self.entry_id.config(state="readonly")
        
        self.absence_count.config(text=str(absences))
        self.combo_status.set("Present")  # Default to Present

    def capture_image(self):
        if not model:
            messagebox.showerror("Error", "Model not found. Face recognition disabled.")
            return
            
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera.")
            return

        messagebox.showinfo("Camera", "Press 's' to save the image and close, or 'q' to quit without saving.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Capture Image - Press 's' to Save", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                img_path = "captured_image.jpg"
                cv2.imwrite(img_path, frame)
                cap.release()
                cv2.destroyAllWindows()
                self.classify_from_path(img_path)
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

    def select_and_classify(self):
        if not model:
            messagebox.showerror("Error", "Model not found. Face recognition disabled.")
            return
            
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")],
            title="Select an Image"
        )
        if file_path:
            self.classify_from_path(file_path)

    def classify_from_path(self, file_path):
        try:
            img = Image.open(file_path).convert("RGB")
        except:
            messagebox.showerror("Error", "Could not open image file.")
            return
            
        # Set maximum display dimensions
        max_display_width = 400
        max_display_height = 300
        
        # Calculate the aspect ratio
        width, height = img.size
        aspect_ratio = width / height
        
        # Calculate new dimensions while maintaining aspect ratio
        if width > height:
            new_width = min(width, max_display_width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(height, max_display_height)
            new_width = int(new_height * aspect_ratio)
        
        # Ensure the image doesn't exceed maximum dimensions in either direction
        if new_width > max_display_width:
            new_width = max_display_width
            new_height = int(new_width / aspect_ratio)
        if new_height > max_display_height:
            new_height = max_display_height
            new_width = int(new_height * aspect_ratio)
        
        # Resize for display
        img_display = img.resize((new_width, new_height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img_display)
        
        # Update image label
        self.image_label.config(image=photo)
        self.image_label.image = photo

        # Resize for model processing
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized).astype(np.float32)
        img_array = (img_array / 127.5) - 1
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array)
        class_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_index])

        # Confidence threshold
        min_confidence = 0.95
        
        if confidence >= min_confidence:
            detected_label = class_names[class_index]
            self.detected_name = detected_label
            self.detected_id = student_database.get(detected_label, {}).get("id", "")
            absences = student_database.get(detected_label, {}).get("absences", 0)
            result_text = (
                f"Detected: {self.detected_name}\n"
                f"ID: {self.detected_id if self.detected_id else 'N/A'}\n"
                f"Absences: {absences}\n"
                f"Confidence Score: {confidence:.2%}")
        else:
            self.detected_name = "Unknown"
            self.detected_id = ""
            result_text = (
                "Unknown Person\n"
                "Confidence Score: 0.00%\n"
                "Please use Manage Student detail")
            
        self.result_label.config(text=result_text)
        self.notebook.select(0)  # Switch to face recognition tab

    def mark_attendance(self, status):
        if not hasattr(self, 'detected_name'):
            messagebox.showerror("Error", "No image processed yet")
            return

        if self.detected_name == "Unknown":
            messagebox.showerror("Error", "Unknown person detected. Use Manage Student detail.")
            return

        # Check if attendance already marked for this person in the current session
        for record in attendance_records:
            if record["name"] == self.detected_name:
                messagebox.showerror("Error", f"Attendance for {self.detected_name} has already been marked.")
                return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "name": self.detected_name,
            "id": self.detected_id,
            "status": status,
            "time": timestamp
        }
        attendance_records.append(record)

        # === Update the database based on attendance status ===
        if self.detected_name in student_database:
            if status == "Absent":
                student_database[self.detected_name]["absences"] += 1
            elif status == "Present":
                # Optional: Reset absences or do nothing
                pass
            save_student_database(student_database)
        else:
            # Fallback if somehow student not found in DB
            messagebox.showerror("Error", f"{self.detected_name} not found in the database.")
            return

        # Show consequences (e.g., warnings for too many absences)
        explanation = self.infer_consequence(self.detected_name)
        self.update_display()

        message = f"Attendance marked for {self.detected_name} as {status}."
        if explanation:
            message += f"\n\n{explanation}"
        messagebox.showinfo("Success", message)



    def mark_manual_attendance(self):
        name = self.student_var.get()
        student_info = student_database.get(name, {})
        student_id = student_info.get("id", "")
        status = self.combo_status.get()

        if not name or not student_id or not status:
            messagebox.showerror("Error", "Please fill all fields")
            return

        # Check if this student has already been marked in this session
        for record in attendance_records:
            if record["name"] == name:
                messagebox.showerror("Error", f"Attendance for {name} has already been marked.")
                return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {"name": name, "id": student_id, "status": status, "time": timestamp}
        attendance_records.append(record)

        # Update absence count in database and save
        if status == "Absent":
            if name in student_database:
                student_database[name]["absences"] += 1
                save_student_database(student_database)
                self.absence_count.config(text=str(student_database[name]["absences"]))

        explanation = self.infer_consequence(name)
        self.update_display()

        message = f"Attendance marked for {name} as {status}."
        if explanation:
            message += f"\n\n{explanation}"
        messagebox.showinfo("Success", message)


    def infer_consequence(self, student_name):
        if student_name not in student_database:
            return ""
            
        absences = student_database[student_name]["absences"]
        explanation = ""

        for rule in reversed(rules):
            if absences >= rule["threshold"]:
                explanation = f"Rule Triggered: {rule['rule']}\nAction: {rule['consequence']}\nTotal Absences: {absences}"
                break

        return explanation

    def update_display(self):
        self.attendance_display.delete("1.0", tk.END)
        header = "Time\t\tID\tName\t\tStatus\n" + "-"*60 + "\n"
        self.attendance_display.insert(tk.END, header)
        
        for record in attendance_records:
            line = f"{record['time']}\t{record['id']}\t{record['name']}\t{record['status']}\n"
            self.attendance_display.insert(tk.END, line)

    def show_absence_report(self):
        report_window = tk.Toplevel(self)
        report_window.title("Absence Frequency Report")
        report_window.geometry("600x500")
        report_window.configure(bg="#e6f2ff")
        
        report_frame = tk.Frame(report_window, bg="#e6f2ff", padx=20, pady=20)
        report_frame.pack(fill="both", expand=True)
        
        tk.Label(report_frame, text="Absence Frequency Report", font=("Arial", 16, "bold"), 
                bg="#e6f2ff", fg="#003366").pack(pady=10)
        
        columns = ("ID", "Name", "Absences", "Status")
        tree = ttk.Treeview(report_frame, columns=columns, show="headings", height=15)
        
        tree.heading("ID", text="Student ID")
        tree.heading("Name", text="Student Name")
        tree.heading("Absences", text="Total Absences")
        tree.heading("Status", text="Status")
        
        tree.column("ID", width=100, anchor="center")
        tree.column("Name", width=200, anchor="center")
        tree.column("Absences", width=100, anchor="center")
        tree.column("Status", width=150, anchor="center")
        
        for student_name, student_info in student_database.items():
            absences = student_info["absences"]
            status = ""
            
            for rule in reversed(rules):
                if absences >= rule["threshold"]:
                    status = rule["consequence"]
                    break
            
            tree.insert("", "end", values=(student_info["id"], student_name, absences, status))
        
        tree.pack(fill="both", expand=True, pady=10)
        
        scrollbar = ttk.Scrollbar(report_frame, orient="vertical", command=tree.yview)
        scrollbar.pack(side="right", fill="y")
        tree.configure(yscrollcommand=scrollbar.set)
        
        btn_close = tk.Button(
            report_frame,
            text="Close",
            font=("Arial", 12),
            bg="#003366",
            fg="#ffffff",
            command=report_window.destroy
        )
        btn_close.pack(pady=10)



# Run the application
if __name__ == "__main__":
    app = IAESApp()
    app.mainloop()