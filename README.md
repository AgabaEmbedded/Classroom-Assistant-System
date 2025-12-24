# Classroom Assistant

**An AI-powered Classroom Management System** that automates student attendance using face recognition and monitors classroom emotions in real-time.

Built with **Streamlit**, **DeepFace**, **MTCNN**, and **TensorFlow**, this app streamlines attendance tracking, detects impersonation during exams, and provides insights into student engagement through emotion analysis.

![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg?style=flat&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![DeepFace](https://img.shields.io/badge/DeepFace-latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## OBJECTIVES:
* To obtain and preprocess emotion dataset.
* To build and train emotion recognition system based on the collected data.
* To development attendance and emotion management system based on the model in 
* To integrate sub-systems developed into a fully functional system.
* To evaluate the system performance using accuracy, precision, recall and Response time.

## Features

- **Automated Attendance Marking**  
  Real-time face detection and recognition during class sessions using images streamed from a Raspberry Pi camera.

- **Emotion Analysis**  
  Detects dominant emotions (anger, calm, disgust, fear, happy, sadness, surprise) across the classroom and visualizes percentages with charts.

- **Exam Mode with Impersonation Detection**  
  Restricts access to students meeting 75% attendance threshold and flags unrecognized faces as "Impersonation".

- **Student Management**  
  Enroll/remove students with face images (via upload or local webcam), organized by courses.

- **Course Management**  
  Create new courses and download attendance records as CSV files.

- **Raspberry Pi Integration**  
  Receives live classroom images over a socket connection from a Raspberry Pi.

## Screenshots

*(Add actual screenshots here once available – recommended: Homepage, Enroll Student, Start Class with emotion chart, Exam Mode)*

## Tech Stack

- **Frontend/UI**: Streamlit
- **Face Detection**: MTCNN
- **Face Recognition**: DeepFace (Facenet / Facenet512)
- **Emotion Recognition**: Custom TensorFlow Keras model (`Latest FER.keras`)
- **Image Processing**: OpenCV, PIL
- **Data Handling**: Pandas, JSON
- **Networking**: Python sockets (for Raspberry Pi image streaming)

## Installation & Setup

### Prerequisites
- Python 3.9 or higher
- A webcam (for enrollment) or Raspberry Pi with camera (for live capture)
- Pre-trained emotion model: `Latest FER.keras` (place in project root)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/classroom-assistant.git
   cd classroom-assistant
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install streamlit tensorflow deepface opencv-python mtcnn pandas matplotlib pillow
   ```

4. **Folder Structure Setup**
   The app expects:
   ```
   student_data/
   ├── attendance/          # CSV files per course
   ├── course list.json     # List of courses
   └── faces/               # Subfolders per course with student face images
   ```
   Run the app once – it will create default folders and files automatically.

5. **(Optional) Raspberry Pi Setup**
   - Run a simple server script on your Raspberry Pi to stream images to IP `192.168.43.230:8000`.
   - Update `raspi_ip` in the code if needed.

### Run the App
```bash
streamlit run main.py
```

Open your browser at `http://localhost:8501`.

## 🎮 Usage Guide

1. **Homepage** – Create new courses or download attendance CSVs.
2. **Enroll Student** – Add student details and capture/upload face image.
3. **Remove Student** – Delete a student and their face data.
4. **Start Class** – Select course, set duration, and begin real-time attendance + emotion monitoring.
5. **Exam Mode** – Only eligible students (≥75% attendance) are marked; others flagged.

## Notes & Limitations

- Requires good lighting and frontal faces for accurate recognition.
- Raspberry Pi streaming uses a hotspot connection (update IP/port as needed).
- Emotion model is custom-trained; accuracy depends on training data.
- For production use, consider security enhancements (e.g., encrypted connections).

## Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests with improvements

Please follow standard GitHub flow: fork → branch → PR.

## License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

---

**Made with ❤️ for smarter classrooms**  
If you find this project useful, give it a ⭐ on GitHub!

Contact sundayabraham81@gmail.com for more information and collaboration.
