Emotion Detection using OpenCV (FER+ Model)

This is a Python project that detects human emotions in real-time using a webcam. It uses a pre-trained FER+ ONNX model and OpenCV's deep learning module (cv2.dnn) for fast and lightweight emotion recognition. No PyTorch or TensorFlow installation is needed.

Features:

Detects 8 emotions: Neutral, Happy, Surprise, Sad, Angry, Disgust, Fear, Contempt

Real-time webcam emotion detection

Draws bounding box around detected faces

Displays predicted emotion with confidence score

Automatic model download if missing

Technologies Used:

Python 3

OpenCV (cv2)

NumPy

FER+ ONNX pre-trained model

Project Structure:
Emotion_Detection/
├── emotion_detection.py # Main script
├── emotion-ferplus-8.onnx # Model (auto-downloaded)
└── README.txt

How to Run:

Make sure Python 3 is installed.

Install required packages:
pip install opencv-python numpy

Run the script:
python emotion_detection.py

The script will automatically download the FER+ ONNX model if it doesn’t exist locally.

A window will open showing your webcam feed with detected faces and emotions.

Press 'q' to quit.

How It Works:

Checks if emotion-ferplus-8.onnx exists, otherwise downloads it.

Uses OpenCV Haar Cascade to detect faces.

Each detected face is preprocessed (grayscale, resized to 64x64, normalized).

FER+ model predicts emotions.

Emotion label and confidence are displayed above each face in real-time.

Supported Emotions:

Neutral

Happy

Surprise

Sad

Angry

Disgust

Fear

Contempt

Purpose:

Demonstrate real-time emotion recognition using computer vision

Practice OpenCV DNN module and ONNX model integration

Build a lightweight and efficient AI project without heavy ML frameworks

Future Improvements:

Add multiple face tracking and simultaneous emotion detection

Save emotion logs with timestamps

Integrate with voice assistants or chatbots

Improve accuracy with more advanced models

Author: Mohsin Atta
