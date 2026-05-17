## Intelligent Attendance Evaluation System(Mini Project for AI)

# 🤖 AI-Based Smart Attendance System

## 📌 Overview
The **AI-Based Smart Attendance System** is a Python desktop application that automates student attendance using face recognition. It replaces traditional manual tracking methods, reducing errors, preventing misuse, and improving administrative efficiency.

The system integrates **computer vision, machine learning, and rule-based logic** to provide an intelligent attendance solution.

---

## 🎯 Objectives
- Automate attendance using face recognition
- Track student presence and absence records
- Enforce attendance policies using rule-based logic
- Provide an admin interface for student management
- Build an easy-to-use and extendable system

---

## ⚙️ Features

### 🧠 Face Recognition
- Identifies students using trained model (`keras_model.h5`)
- Uses OpenCV, PIL, and TensorFlow/Keras
- Prevents duplicate attendance marking in the same session

### 📝 Attendance Tracking
- Marks students as present/absent
- Tracks total presences and absences
- Stores data in `student_database.json`

### ⚖️ Rule-Based System
- Applies consequences based on absence thresholds:
  - 3 absences → Warning
  - 5 absences → Meeting required
  - 7 absences → Disciplinary action

### 👨‍💻 Admin Management
- Add new students
- Delete existing students
- Automatically syncs UI with database

### 🖥️ User Interface
- Built with Tkinter
- Tabs include:
  - Face Recognition
  - Manual Entry
  - Student Management

---



<img width="875" height="493" alt="image" src="https://github.com/user-attachments/assets/c89f0668-6f7c-43f0-a940-597660b0a61e" />

<br><br>


![ai_attendance1](https://github.com/user-attachments/assets/9c62157d-23a2-4bca-a2cb-5d8e864104cf)


<br><br>


![ai_attendance3](https://github.com/user-attachments/assets/8a8d44d0-324d-4a64-901b-c5f77d4208de)

<br><br>

<img width="1313" height="907" alt="ai_attendance4" src="https://github.com/user-attachments/assets/f615768e-411a-4808-a8cc-1a885fd04851" />


## 🎥 Presentation Video

[![Watch the video](https://img.youtube.com/vi/cDHOgv72i_k/0.jpg)](https://youtu.be/cDHOgv72i_k)
