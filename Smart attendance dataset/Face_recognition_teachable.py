import os
import json
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# === Load or create student database ===
def load_student_database():
    if os.path.exists("student_database.json"):
        try:
            with open("student_database.json", "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Warning: student_database.json is corrupt. Loading default database.")
    # Default fallback
    return {
        "Abir": {"id": "1", "absences": 3},
        "Fahim": {"id": "2", "absences": 1},
        "Hemel": {"id": "3", "absences": 0},
        "Nipa": {"id": "4", "absences": 8}
    }

def save_student_database(db):
    with open("student_database.json", "w") as f:
        json.dump(db, f, indent=2)

# === Rules for consequences based on absences ===
rules = [
    {"threshold": 3, "consequence": "Warning"},
    {"threshold": 5, "consequence": "Meeting with supervisor"},
    {"threshold": 7, "consequence": "Disciplinary action"}
]

# === Load model ===
if not os.path.exists("keras_model.h5"):
    raise FileNotFoundError("Error: 'keras_model.h5' not found.")
model = load_model("keras_model.h5", compile=False)

# === Load labels ===
if not os.path.exists("labels.txt"):
    raise FileNotFoundError("Error: 'labels.txt' not found.")
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# === Load captured image ===
image_path = "29.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Error: '{image_path}' not found.")
image = Image.open(image_path).convert("RGB")
image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

# === Preprocess image for prediction ===
image_array = np.asarray(image).astype(np.float32)
normalized_image_array = (image_array / 127.5) - 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
data[0] = normalized_image_array

# === Perform prediction ===
prediction = model.predict(data)
top_index = np.argmax(prediction)
max_confidence = prediction[0][top_index]
min_confidence = 0.9750

# === Process prediction result ===
student_database = load_student_database()
raw_label = class_names[top_index]
predicted_name = " ".join(raw_label.split()[1:])  # Skip index (e.g., "0 Hemel" -> "Hemel")
student = student_database.get(predicted_name)

# === Decision logic ===
if max_confidence >= min_confidence and student:
    student_id = student["id"]
    absences = student["absences"]

    # Find applicable consequence
    consequence = next((rule["consequence"] for rule in rules if absences >= rule["threshold"]), "None")

    result_text = (
        f"Detected: {predicted_name}\n"
        f"ID: {student_id}\n"
        f"Absences: {absences}\n"
        f"Confidence: {max_confidence:.2%}\n"
        f"Consequence: {consequence}"
    )
else:
    result_text = (
        "Unknown Person\n"
        f"Confidence: {max_confidence:.2%}\n"
        "No matching record found in student database."
    )

# === Output the result ===
print(result_text)
