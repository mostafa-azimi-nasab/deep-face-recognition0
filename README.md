# Deep Face Recognition

A real-time face recognition project developed with Python, OpenCV,
MTCNN and FaceNet.

The system detects faces using MTCNN and extracts face embeddings using
the pretrained InceptionResnetV1 FaceNet model.

## Features

- Real-time face detection
- Deep-learning-based face recognition
- Face registration using a webcam
- Multiple face support
- Unknown-person detection
- Face embedding database
- CPU and CUDA GPU support
- Modular Python project structure
- Unit tests

## Technologies

- Python
- PyTorch
- OpenCV
- FaceNet
- MTCNN
- NumPy

## Project Structure

```text
deep-face-recognition/
├── app.py
├── enroll.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── database.py
│   └── face_engine.py
├── data/
│   ├── known_faces/
│   └── embeddings/
└── tests/
    └── test_database.py