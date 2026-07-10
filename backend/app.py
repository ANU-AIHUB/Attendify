print("ATTENDIFY APP STARTED")

import csv
from email.mime import image
from fileinput import filename
from multiprocessing.reduction import duplicate
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import cv2
from flask import Flask, jsonify, request
from pathlib import Path
import subprocess
from deepface import DeepFace
import json
import numpy as np
from scipy.spatial.distance import cosine
from werkzeug.utils import secure_filename
import numpy as np
import shutil


app = Flask(__name__)
LABEL_MAPPING_PATH = Path("backend/label_mapping.csv")
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "attendify.db"
CASCADE_PATH = BASE_DIR / "models" / "haarcascade_frontalface_default.xml"
TRAINER_PATH = BASE_DIR / "trainer.yml"
CAPTURED_FACE_PATH = BASE_DIR / "captured_face.jpg"
DEFAULT_STUDENT_ID = "0112AL231016"
TIME_INPUT_FORMATS = (
    "%H:%M",
    "%H:%M:%S",
    "%I:%M %p",
    "%I:%M:%S %p",
)


def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column(cursor, table_name, column_name, column_definition):
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if column_name not in existing_columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_definition}")


def normalize_time_value(value):
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    for time_format in TIME_INPUT_FORMATS:
        try:
            return datetime.strptime(text, time_format).strftime("%H:%M")
        except ValueError:
            continue

    return None

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def normalize_lecture_times(conn):
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, start_time, end_time
        FROM lectures
        """
    )

    updates = []
    for lecture_id, start_time, end_time in cursor.fetchall():
        normalized_start = normalize_time_value(start_time)
        normalized_end = normalize_time_value(end_time)

        if normalized_start and normalized_end and (
            normalized_start != start_time or normalized_end != end_time
        ):
            updates.append((normalized_start, normalized_end, lecture_id))

    if updates:
        cursor.executemany(
            """
            UPDATE lectures
            SET start_time = ?, end_time = ?
            WHERE id = ?
            """,
            updates,
        )
        conn.commit()


def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        student_id TEXT PRIMARY KEY,
        full_name TEXT NOT NULL,
        branch TEXT NOT NULL,
        semester INTEGER NOT NULL,
        section TEXT NOT NULL,
        email TEXT,
        mobile TEXT,
        password TEXT NOT NULL
    )
    """)

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS faculty (
            faculty_id TEXT PRIMARY KEY,
            name TEXT,
            password TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS lectures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT,
            start_time TEXT,
            end_time TEXT,
            faculty_id TEXT,
            latitude REAL,
            longitude REAL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            lecture_id INTEGER,
            date TEXT,
            time TEXT
        )
        """
    )

    ensure_column(cursor, "students", "password", "password TEXT")
    ensure_column(cursor, "lectures", "faculty_id", "faculty_id TEXT")
    ensure_column(cursor, "lectures", "latitude", "latitude REAL")
    ensure_column(cursor, "lectures", "longitude", "longitude REAL")
    ensure_column(cursor, "lectures", "radius", "radius REAL DEFAULT 50")
    ensure_column(cursor, "attendance", "lecture_id", "lecture_id INTEGER")
    ensure_column(cursor, "lectures", "branch", "branch TEXT")
    ensure_column(cursor, "lectures", "semester", "semester INTEGER")
    ensure_column(cursor, "lectures", "section", "section TEXT")
    ensure_column(cursor, "lectures", "lecture_date", "lecture_date TEXT")

    normalize_lecture_times(conn)

    conn.commit()
    conn.close()


initialize_database()

def save_face_embedding(student_id, image_path):

    embedding = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet",
        enforce_detection=True
    )

    vector = embedding[0]["embedding"]

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO face_embeddings
        (student_id, embedding)
        VALUES (?, ?)
    """, (
        student_id,
        json.dumps(vector)
    ))

    conn.commit()
    conn.close()

    return True


@app.route("/students")
def get_students():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT student_id, full_name
        FROM students
        """
    )

    records = cursor.fetchall()
    conn.close()

    return {
        "students": [
            {
                "student_id": row[0],
                "name": row[1]
            }
            for row in records
        ]
    }

@app.route("/attendance/subject/<subject>")
def get_subject_attendance(subject):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT students.name, attendance.date, attendance.time
        FROM attendance
        JOIN lectures
        ON attendance.lecture_id = lectures.id
        JOIN students
        ON attendance.student_id = students.student_id
        WHERE lectures.subject = ?
        """,
        (subject,)
    )

    records = cursor.fetchall()
    conn.close()

    return {
        "attendance": [
            {
                "name": row[0],
                "date": row[1],
                "time": row[2]
            }
            for row in records
        ]
    }


@app.route("/add-student", methods=["POST"])
def add_student():
    data = request.get_json(silent=True) or {}
    student_id = data.get("student_id")
    name = data.get("name")

    if not student_id or not name:
        return {"success": False, "message": "student_id and name are required"}, 400

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO students (student_id, name)
        VALUES (?, ?)
        """,
        (student_id, name)
    )

    conn.commit()
    conn.close()

    return {"success": True, "message": "Student added successfully"}


@app.route("/add-student-test")
def add_student_test():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR IGNORE INTO students (student_id, name)
        VALUES (?, ?)
        """,
        ("0112AL231017", "Test Student")
    )

    conn.commit()
    conn.close()
    return {"success": True, "message": "Student added successfully"}


@app.route("/login/<student_id>/<password>")
def login(student_id, password):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT full_name
        FROM students
        WHERE student_id = ?
        AND password = ?
        """,
        (student_id, password)
    )

    student = cursor.fetchone()
    conn.close()

    if student:
        return {
            "success": True,
            "name": student[0]
        }

    return {
        "success": False,
        "message": "Invalid credentials"
    }

def get_student_id_from_label(label):
    mapping = []
    if LABEL_MAPPING_PATH.exists():
        with open(LABEL_MAPPING_PATH, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                mapping.append(row)

    for entry in mapping:
        if str(entry["label"]) == str(label):
            return entry["student_id"]

    return None


@app.route("/mark-attendance", methods=["POST"])
def mark_attendance():

    if "image" not in request.files:
        return {
            "success": False,
            "message": "No image uploaded"
        }

    image = request.files["image"]

    login_student_id = request.form.get("student_id")
    lecture_id = request.form.get("lecture_id")

    filename = f"{login_student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

    file_bytes = np.frombuffer(image.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "success": False,
            "message": "Invalid image"
        }

    print("ATTENDANCE IMAGE SIZE:", img.shape)

    if img is None:
        return {
            "success": False,
            "message": "Invalid image file"
        }

    student_latitude = float(
        request.form.get("latitude", 0)
    )

    student_longitude = float(
        request.form.get("longitude", 0)
    )

    login_student_id = request.form.get(
        "student_id"
    )

    lecture_id = request.form.get(
        "lecture_id"
    )

    print("Student Lat:", student_latitude)
    print("Student Lng:", student_longitude)
    print("Login Student ID:", login_student_id)
    print("Lecture ID:", lecture_id)

    conn = get_db_connection()
    cursor = conn.cursor()

    try:

        embedding = DeepFace.represent(
    img_path=image,
    model_name="Facenet",
    detector_backend="retinaface",
    enforce_detection=True,
    align=True
)

        current_embedding = np.array(
            embedding[0]["embedding"]
        )
        print("ATTENDANCE LENGTH:", len(current_embedding))
        print("ATTENDANCE FIRST 5:", current_embedding[:5])
        print("Attendance Embedding Length:", len(current_embedding))
        print("Attendance Min:", np.min(current_embedding))
        print("Attendance Max:", np.max(current_embedding))
        print("Attendance Mean:", np.mean(current_embedding))

        cursor.execute("""
            SELECT student_id, embedding
            FROM face_embeddings
        """)

        rows = cursor.fetchall()

        if len(rows) == 0:
            return {
                "success": False,
                "message": "No face embeddings found"
            }

        best_student = None
        best_distance = 999

        for row in rows:

            db_student_id = row[0]

            db_embedding = np.array(
                json.loads(row[1])
            )

            distance = cosine(
                current_embedding,
                db_embedding
            )

            print(
                "Compare:",
                db_student_id,
                "Distance:",
                distance
            )

            if distance < best_distance:

                best_distance = distance
                best_student = db_student_id

        print("Best Match:", best_student)
        print("Best Distance:", best_distance)

        if best_student is None:

            return {
                "success": False,
                "message": "Face not recognized"
            }

        if best_distance > 0.40:

            return {
                "success": False,
                "message": "Face not recognized"
            }

        student_id = best_student

        if str(student_id) != str(login_student_id):

            return {
                "success": False,
                "message": "Face does not match logged-in student"
            }

        cursor.execute(
            """
            SELECT full_name
            FROM students
            WHERE student_id = ?
            """,
            (student_id,)
        )

        student = cursor.fetchone()

        if not student:

            return {
                "success": False,
                "message": "Student not found"
            }

        student_name = student[0]

        current_time = datetime.now().strftime("%H:%M")
        today = datetime.now().strftime("%Y-%m-%d")

        cursor.execute(
            """
            SELECT id, latitude, longitude, radius
            FROM lectures
            WHERE id = ?
            """,
            (lecture_id,)
        )

        lecture = cursor.fetchone()

        if not lecture:

            return {
                "success": False,
                "message": "Lecture not found"
            }

        lecture_id, lecture_lat, lecture_lng, lecture_radius = lecture

        print("LECTURE LOCATION:", lecture_lat, lecture_lng)
        distance = calculate_distance(
            student_latitude,
            student_longitude,
            lecture_lat,
            lecture_lng
        )

        print("Distance:", distance)

        if distance > lecture_radius:

            return {
                "success": False,
                "message": "Outside lecture radius"
            }

        cursor.execute(
            """
            SELECT id
            FROM attendance
            WHERE student_id = ?
            AND lecture_id = ?
            AND date = ?
            """,
            (
                student_id,
                lecture_id,
                today
            )
        )

        record = cursor.fetchone()

        if record:

            return {
                "success": False,
                "message": "Attendance already marked"
            }

        cursor.execute(
            """
            INSERT INTO attendance
            (
                student_id,
                lecture_id,
                date,
                time
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                student_id,
                lecture_id,
                today,
                current_time
            )
        )

        conn.commit()

        print({
    "success": True,
    "name": student_name,
    "message": "Attendance Marked Successfully"
})

        return {
            "success": True,
            "name": student_name,
            "message": "Attendance Marked Successfully"
        }

    except Exception as e:

        print("ERROR:", str(e))

        return {
            "success": False,
            "message": str(e)
        }

    finally:
        conn.close()

@app.route("/create-lecture", methods=["POST"])
def create_lecture():
    data = request.get_json(silent=True) or {}

    subject = data.get("subject")
    subject_code = data.get("subject_code", "").strip().upper()
    branch = data.get("branch", "").strip().upper()

    semester = data.get("semester")
    section = data.get("section", "").strip().upper()
    lecture_date = data.get("lecture_date")
    raw_start_time = data.get("start_time")
    raw_end_time = data.get("end_time")
    start_time = normalize_time_value(raw_start_time)
    end_time = normalize_time_value(raw_end_time)
    faculty_id = data.get("faculty_id")
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    radius = data.get("radius", 50)

    if (
    not subject
    or not subject_code
    or not start_time
    or not end_time
    or not faculty_id
    or not branch
    or not semester
    or not section
    or not lecture_date
    ):
    
        return {
        "success": False,
        "message": "Missing required fields"
        }, 400

    if raw_start_time is not None and start_time is None:
        return {
            "success": False,
            "message": "Invalid start_time format"
        }, 400

    if raw_end_time is not None and end_time is None:
        return {
            "success": False,
            "message": "Invalid end_time format"
        }, 400

    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT id
        FROM lectures
        WHERE branch = ?
        AND semester = ?
        AND section = ?
        AND lecture_date = ?
        AND start_time < ?
        AND end_time > ?
        """,
        (
            branch,
            semester,
            section,
            lecture_date,
            end_time,
            start_time
        )
    )

    existing_lecture = cursor.fetchone()
    print("Existing Lecture:", existing_lecture)

    if existing_lecture:
        conn.close()
        return {
            "success": False,
            "message": "Lecture already exists for this Branch, Semester and Section during this time"
        }, 400

    cursor.execute(
        """
       INSERT INTO lectures
        (
            subject,
            subject_code,
            start_time,
            end_time,
            faculty_id,
            latitude,
            longitude,
            radius,
            branch,
            semester,
            section,
            lecture_date
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            subject,
            subject_code,
            start_time,
            end_time,
            faculty_id,
            latitude,
            longitude,
            radius,
            branch,
            semester,
            section,
            lecture_date
        )
    )

    conn.commit()
    conn.close()

    return {
        "success": True,
        "message": "Lecture Created"
    }


@app.route("/student-report")
def student_report_all():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT students.name,
               lectures.subject,
               attendance.date,
               attendance.time
        FROM attendance
        JOIN students
        ON attendance.student_id = students.student_id
        JOIN lectures
        ON attendance.lecture_id = lectures.id
        """
    )

    records = cursor.fetchall()
    conn.close()

    return {
        "reports": [
            {
                "name": row[0],
                "subject": row[1],
                "date": row[2],
                "time": row[3]
            }
            for row in records
        ]
    }


@app.route("/export-csv")
def export_all_csv():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT students.name,
               lectures.subject,
               attendance.date,
               attendance.time
        FROM attendance
        JOIN students
        ON attendance.student_id = students.student_id
        JOIN lectures
        ON attendance.lecture_id = lectures.id
        """
    )

    records = cursor.fetchall()

    with open(
        BASE_DIR / "attendance_report.csv",
        "w",
        newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "Student",
            "Subject",
            "Date",
            "Time"
        ])
        writer.writerows(records)

    conn.close()

    return {
        "success": True,
        "message": "CSV Exported"
    }


@app.route("/export-csv/<faculty_id>")
def export_faculty_csv(faculty_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT students.name,
               lectures.subject,
               attendance.date,
               attendance.time
        FROM attendance
        JOIN students
        ON attendance.student_id = students.student_id
        JOIN lectures
        ON attendance.lecture_id = lectures.id
        WHERE lectures.faculty_id = ?
        """,
        (faculty_id,)
    )

    records = cursor.fetchall()

    with open(
        BASE_DIR / f"{faculty_id}_attendance_report.csv",
        "w",
        newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "Student",
            "Subject",
            "Date",
            "Time"
        ])
        writer.writerows(records)

    conn.close()

    return {
        "success": True,
        "message": "CSV Exported"
    }
@app.route('/register-faculty', methods=['POST'])
def register_faculty():
    data = request.json

    faculty_id = data['faculty_id'].strip().upper()
    name = data['name'].strip().upper()
    password = data['password']

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO faculty
            (faculty_id, name, password)
            VALUES (?, ?, ?)
        """, (
           faculty_id, 
	name, 
	password
        ))

        conn.commit()

        return {
            "success": True,
            "message": "faculty Registered Successfully"
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }

    finally:
        conn.close()
    
@app.route("/faculty-login/<faculty_id>/<password>")
def faculty_login(faculty_id, password):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT name
        FROM faculty
        WHERE faculty_id = ?
        AND password = ?
        """,
        (faculty_id.strip().upper(), password)
    )

    faculty = cursor.fetchone()
    conn.close()

    if faculty:
        return {
            "success": True,
            "name": faculty[0]
        }

    return {
        "success": False,
        "message": "Invalid Faculty Credentials"
    }


@app.route("/student-report/<faculty_id>")
def student_report_by_faculty(faculty_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT students.full_name,
               lectures.subject,
               attendance.date,
               attendance.time
        FROM attendance
        JOIN students
        ON attendance.student_id = students.student_id
        JOIN lectures
        ON attendance.lecture_id = lectures.id
        WHERE lectures.faculty_id = ?
        """,
        (faculty_id,)
    )

    records = cursor.fetchall()
    conn.close()

    return {
        "reports": [
            {
                "name": row[0],
                "subject": row[1],
                "date": row[2],
                "time": row[3]
            }
            for row in records
        ]
    }
@app.route('/register-student', methods=['POST'])
def register_student():
    data = request.json

    student_id = data['student_id'].strip().upper()
    full_name = data['full_name'].strip().upper()
    branch = data['branch'].strip().upper()
    semester = data['semester']
    section = data['section'].strip().upper()
    email = data['email'].strip().upper()
    mobile = data["mobile"].strip()

    if not mobile.isdigit() or len(mobile) != 10:
        return {
            "success": False,
            "message": "Mobile number must be exactly 10 digits"
        }, 400
    password = data['password']

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO students
            (student_id, full_name, branch,
             semester, section, email,
             mobile, password)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            student_id,
            full_name,
            branch,
            semester,
            section,
            email,
            mobile,
            password
        ))

        conn.commit()

        return {
            "success": True,
            "message": "Student Registered Successfully"
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }

    finally:
        conn.close()

def is_duplicate_face(new_embedding, threshold=0.40):

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT student_id, embedding
        FROM face_embeddings
    """)

    rows = cursor.fetchall()
    conn.close()

    new_embedding = np.array(new_embedding)

    print("Total embeddings:", len(rows))

    for row in rows:

        student_id = row[0]
        stored_embedding = np.array(json.loads(row[1]))

        distance = cosine(new_embedding, stored_embedding)

        print(
            "Checking:",
            student_id,
            "Distance:",
            distance
        )

        if distance <= threshold:
            return True, student_id

    return False, None

@app.route('/upload-face', methods=['POST'])
def upload_face():

    student_id = request.form.get("student_id")

    image = request.files.get("image")


    if not student_id or not image:

        return jsonify({
            "success": False,
            "message": "Missing data"
        })

    save_dir = os.path.join(
        "dataset",
        student_id
    )

    os.makedirs(save_dir, exist_ok=True)

    image_path = os.path.join(
        save_dir,
        image.filename
    )

    image.save(image_path)
    image = cv2.imread(image_path)

    if image is not None:
        print("REGISTER IMAGE SIZE:", image.shape)
    else:
        print("Could not read registration image")

    try:

        embedding = DeepFace.represent(
    img_path=image_path,
    model_name="Facenet",
    detector_backend="retinaface",
    enforce_detection=True,
    align=True
)

        vector = embedding[0]["embedding"]
        print("Registration Embedding Length:", len(vector))
        print("Registration Min:", min(vector))
        print("Registration Max:", max(vector))
        print("Registration Mean:", np.mean(vector))
        duplicate, existing_student = is_duplicate_face(vector)
        print("Duplicate:", duplicate)
        print("Existing Student:", existing_student)
        print("REGISTER LENGTH:", len(vector))
        print("REGISTER FIRST 5:", vector[:5])

        if duplicate:

            # Delete uploaded image
            if os.path.exists(image_path):
                os.remove(image_path)

            # Delete dataset folder
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)

            # Delete student record
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM students WHERE student_id = ?",
                (student_id,)
            )

            conn.commit()
            conn.close()

            return jsonify({
                "success": False,
                "message": f"This face is already registered with Student ID {existing_student}"
            }), 409    

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO face_embeddings
            (student_id, embedding)
            VALUES (?, ?)
        """, (
            student_id,
            json.dumps(vector)
        ))

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "Face embedding saved"
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        })
    

    
@app.route('/train-model', methods=['POST'])
def train_model_api():

    try:

        result = subprocess.run(
        [
        "python",
        "train_model.py"
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
        )


        return jsonify({
            "success": True,
            "output": result.stdout
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
@app.route("/student/live-lectures/<student_id>", methods=["GET"])
def get_live_lectures(student_id):

    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT branch, semester, section
        FROM students
        WHERE student_id = ?
    """, (student_id,))

    student = cursor.fetchone()
    print("Student Data:")
    print(dict(student))

    if not student:
        conn.close()
        return {
            "success": False,
            "message": "Student not found"
        }, 404

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M")
    print("Today:", today)
    print("Current Time:", current_time)    

    cursor.execute("""
        SELECT *
        FROM lectures
        WHERE branch = ?
        AND semester = ?
        AND section = ?
        AND lecture_date = ?
        AND start_time <= ?
        AND end_time >= ?
    """, (
        student["branch"],
        student["semester"],
        student["section"],
        today,
        current_time,
        current_time
    ))

    lectures = [dict(row) for row in cursor.fetchall()]
    print("Lectures Found:", len(lectures))

    conn.close()

    return {
        "success": True,
        "lectures": lectures
    }
@app.route("/student/attendance-summary/<student_id>", methods=["GET"])
def attendance_summary(student_id):

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM attendance
        WHERE student_id = ?
    """, (student_id,))

    total_attended = cursor.fetchone()[0]

    conn.close()

    return {
        "success": True,
        "total_attended": total_attended
    }

@app.route("/faculty/lecture-summary/<faculty_id>", methods=["GET"])
def faculty_lecture_summary(faculty_id):

    faculty_id = faculty_id.strip().upper()

    conn = get_db_connection()
    cursor = conn.cursor()

    print("Faculty ID Received:", faculty_id)

    cursor.execute("""
    SELECT subject,
           subject_code,
           COUNT(*) AS total
    FROM lectures
    WHERE faculty_id = ?
    GROUP BY subject, subject_code
""", (faculty_id,))

    rows = cursor.fetchall()
    print(rows)

    data = []

    for row in rows:
        data.append({
            "subject": row[0],
            "subject_code": row[1],
            "total": row[2]
        })

    conn.close()

    return {
        "success": True,
        "subjects": data
    }


@app.route("/student/attendance-percentage/<student_id>")
def attendance_percentage(student_id):

    conn = get_db_connection()
    cursor = conn.cursor()

    # Attendance marked by student
    cursor.execute("""
        SELECT COUNT(*)
        FROM attendance
        WHERE student_id = ?
    """, (student_id,))

    attended = cursor.fetchone()[0]

    # Get student's branch, semester, section
    cursor.execute("""
        SELECT branch, semester, section
        FROM students
        WHERE student_id = ?
    """, (student_id,))

    student = cursor.fetchone()

    if not student:

        conn.close()

        return {
            "success": False,
            "message": "Student not found"
        }

    branch, semester, section = student

    # Count only lectures eligible for this student
    cursor.execute("""
        SELECT COUNT(*)
        FROM lectures
        WHERE branch = ?
        AND semester = ?
        AND section = ?
    """, (branch, semester, section))

    total_lectures = cursor.fetchone()[0]

    percentage = 0

    if total_lectures > 0:
        percentage = round(
            (attended * 100) / total_lectures,
            2
        )

    conn.close()

    return {
        "success": True,
        "attended": attended,
        "total_lectures": total_lectures,
        "percentage": percentage,
        "branch": branch,
        "semester": semester,
        "section": section
    }
@app.route("/student/subject-attendance/<student_id>")
def subject_attendance(student_id):

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT branch, semester, section
        FROM students
        WHERE student_id = ?
    """, (student_id,))

    student = cursor.fetchone()

    if not student:

        conn.close()

        return {
            "success": False,
            "message": "Student not found"
        }

    branch, semester, section = student

    cursor.execute("""
        SELECT DISTINCT subject
        FROM lectures
        WHERE branch = ?
        AND semester = ?
        AND section = ?
    """, (branch, semester, section))

    subjects = cursor.fetchall()

    result = []

    for row in subjects:

        subject = row[0]

        # Total lectures of this subject
        cursor.execute("""
            SELECT COUNT(*)
            FROM lectures
            WHERE subject = ?
            AND branch = ?
            AND semester = ?
            AND section = ?
        """, (subject, branch, semester, section))

        total = cursor.fetchone()[0]

        # Attended lectures of this subject
        cursor.execute("""
            SELECT COUNT(*)
            FROM attendance a
            JOIN lectures l
            ON a.lecture_id = l.id
            WHERE a.student_id = ?
            AND l.subject = ?
        """, (student_id, subject))

        attended = cursor.fetchone()[0]

        percentage = 0

        if total > 0:
            percentage = round(
                (attended * 100) / total,
                2
            )

        result.append({
            "subject": subject,
            "attended": attended,
            "total": total,
            "percentage": percentage
        })

    conn.close()

    return {
        "success": True,
        "subjects": result
    }
@app.route("/student/profile/<student_id>")
def student_profile(student_id):

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT full_name,
               student_id,
               branch,
               semester,
               section,
               email,
               mobile
        FROM students
        WHERE student_id = ?
    """, (student_id,))

    row = cursor.fetchone()

    conn.close()

    if not row:
        return {"success": False}

    return {
        "success": True,
        "full_name": row[0],
        "student_id": row[1],
        "branch": row[2],
        "semester": row[3],
        "section": row[4],
        "email": row[5],
        "mobile": row[6]
    }

@app.route("/student/upload-profile-image", methods=["POST"])
def upload_profile_image():

    student_id = request.form.get("student_id")

    if "image" not in request.files:
        return {"success": False, "message": "No image uploaded"}

    image = request.files["image"]

    if image.filename == "":
        return {"success": False, "message": "Empty image"}

    filename = secure_filename(f"{student_id}.jpg")

    save_path = os.path.join("profile_images", filename)

    image.save(save_path)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE students
        SET profile_image = ?
        WHERE student_id = ?
    """, (filename, student_id))

    conn.commit()
    conn.close()

    return {
        "success": True,
        "filename": filename
    }

@app.route('/test-train')
def test_train():
    return train_model_api()



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
