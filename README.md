# MNIST--digit--recognizer
This project implements a handwritten digit recognition system using the MNIST dataset and a Convolutional Neural Network (CNN) built in TensorFlow/Keras. The model classifies grayscale images of digits (0–9) with high accuracy. Additionally, it is deployed as a Streamlit web application for real-time predictions.

🚀 Features
Preprocessing of MNIST dataset: normalization, train-test split.
CNN model built with TensorFlow/Keras for image classification.
Real-time predictions via Streamlit web interface.

📊 Dataset
MNIST Handwritten Digit Dataset
Contains 70,000 grayscale images of digits (28x28 pixels).
60,000 images for training and 10,000 images for testing.
Directly loaded from tensorflow.keras.datasets.mnist.

⚙️ CNN Architecture
Input layer: 28x28 grayscale image (1 channel)
Convolutional Layer 1: 32 filters, 3x3 kernel, ReLU activation
MaxPooling Layer 1: 2x2 pool size
Convolutional Layer 2: 64 filters, 3x3 kernel, ReLU activation
MaxPooling Layer 2: 2x2 pool size
Flatten Layer: Converts 2D feature maps into 1D vector
Dense Layer: 128 neurons, ReLU activation
Dropout Layer: 0.5 dropout rate (prevents overfitting)
Output Layer: 10 neurons, Softmax activation (for 10 digit classes: 0–9).

⚙️ Requirements
numpy, matplotlib, tensorflow, streamlit, pillow

📈 Results
Test Accuracy: ~99% (depending on training parameters)
Real-time digit predictions via Streamlit app
High generalization on unseen handwritten digits

📌 Future Improvements
Experiment with deeper CNN architectures or ResNet for better generalization.
Integrate real-time drawing interface enhancements in Streamlit (e.g., color customization, multi-digit recognition).
Extend the app to recognize digits in complex backgrounds or scanned documents.

✨ Author
👩‍💻 Developed by PreethiGorantla using Google Colab and Streamlit



