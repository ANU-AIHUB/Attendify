# 🎓 Attendify - Smart Attendance System

Attendify is an AI-powered smart attendance system that uses **Face Recognition** and **GPS-based Geofencing** to automate classroom attendance securely and accurately.

## 🚀 Features

- 👤 Student Registration
- 🔐 Secure Student Login
- 😊 Face Recognition-based Attendance
- 📍 GPS Geofencing (50-meter radius)
- 👨‍🏫 Faculty Dashboard
- 📚 Lecture-wise Attendance
- 📊 Attendance Reports
- 📱 Android Application
- 🌐 Flask Backend API
- 🗄️ SQLite Database

---

## 🛠️ Tech Stack

### Frontend
- Android Studio
- Java
- XML

### Backend
- Python
- Flask

### AI & Machine Learning
- DeepFace
- InsightFace
- OpenCV
- Haar Cascade

### Database
- SQLite

### Tools
- Git
- GitHub
- Postman

---

## 📂 Project Structure

```
Attendify/
│
├── app/
├── backend/
│   ├── app.py
│   ├── models/
│   └── ...
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/ANU-AIHUB/Attendify.git
```

### Navigate to the Project

```bash
cd Attendify
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Virtual Environment

**Windows**

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Backend

```bash
python backend/app.py
```

---

## 📱 How It Works

1. Student registers in the app.
2. Face embeddings are generated and stored.
3. Faculty creates a lecture with GPS location.
4. Student enters classroom.
5. GPS verifies location.
6. Face Recognition verifies identity.
7. Attendance is marked only if both verifications succeed.

---

## 🔒 Security Features

- Face Recognition Authentication
- GPS Geofencing
- Duplicate Face Detection
- Lecture-wise Attendance Validation

---

## 📌 Future Enhancements

- Cloud Database Integration
- QR Code Attendance
- Face Anti-Spoofing
- Email Notifications
- Analytics Dashboard

---

## 👩‍💻 Developer

**Anuradha Verma**

B.Tech (Artificial Intelligence & Machine Learning)

GitHub: https://github.com/ANU-AIHUB

---

## ⭐ If you like this project

Give this repository a ⭐ on GitHub.
