import os
import shutil

# Folder containing your images
SOURCE_DIR = "19CSE213"   # change if needed

# Output folder
DEST_DIR = "New Data/19CSE_Arr"

os.makedirs(DEST_DIR, exist_ok=True)

# Get all jpg files
files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(".jpg")]

# Extract unique roll numbers
roll_numbers = sorted(set(f.split("_")[0] for f in files))

# Map roll number -> student index (s1, s2, ...)
roll_to_student = {
    roll: f"cs_s19{i+1}"
    for i, roll in enumerate(roll_numbers)
}

# Create student folders
for student in roll_to_student.values():
    os.makedirs(os.path.join(DEST_DIR, student), exist_ok=True)

# Move files
for file in files:
    roll = file.split("_")[0]
    student_folder = roll_to_student[roll]

    src = os.path.join(SOURCE_DIR, file)
    dst = os.path.join(DEST_DIR, student_folder, file)

    shutil.copy(src, dst)

print("Reorganization complete.")