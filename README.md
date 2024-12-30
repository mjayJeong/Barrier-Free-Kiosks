# Barrier-Free Kiosks Model

This project presents an accessible kiosk system that includes gesture recognition, facial recognition, eye tracking, and voice recognition. It is designed to cater to diverse users, including individuals with disabilities, older adults, and those with limited digital literacy. The system provides seamless interaction through intuitive user interfaces and intelligent model integration.

---

## Project Overview

### Purpose
The Barrier-Free Kiosks aim to improve digital accessibility by:
- Supporting individuals with disabilities, older adults, and others with limited access to digital technology.
- Providing user-friendly solutions by integrating gesture recognition, facial recognition, eye tracking, and voice recognition.
- Ensuring a smooth and inclusive experience for all users.

---

## Features

### 1. Gesture Recognition
- **Model**: Built using MediaPipe for real-time hand tracking.
- **Data Preparation**:
  - Frames captured from YouTube videos were preprocessed under various conditions (lighting, background, camera angles).
  - Data augmentation techniques like brightness adjustment, shearing, and zoom were applied to create a robust dataset.
- **Implementation**: CNN architecture used for high precision and efficiency in gesture classification.

### 2. Face Recognition
- **Model**: Combines DeepFace and AgeNet using an ensemble method:
  - **DeepFace**: High weight (70%) for precise age estimation.
  - **AgeNet**: Low weight (30%) to handle broader age ranges.
- **Dependencies**:
  - `age_net.caffemodel`
  - `deploy_age.prototxt`
  - `haarcascade_frontalface_default.xml`

### 3. Eye Tracking
- **Model**: Built with MediaPipe FaceMesh for real-time tracking of eye and pupil positions.
- **Calibration**:
  - Users focus on four screen corners (top-left, top-right, bottom-left, bottom-right).
  - Eye movement patterns are recorded and mapped for gaze tracking.
- **Configuration**:
  - Adjust resolution settings in the code for improved performance and accuracy.

### 4. Voice Recognition
- **Comparison**:
  - Whisper and FasterWhisper models were evaluated for speech-to-text conversion.
  - FasterWhisper proved faster and more efficient for longer inputs.
- **Enhancements**:
  - Noise removal for clear audio processing.
  - Matching speech output with predefined word lists using similarity metrics.

---

## Technical Implementation 

### Tools & Libraries
- **Gesture Recognition**: MediaPipe and CNN.
- **Facial Recognition**: DeepFace and AgeNet ensemble.
- **Eye Tracking**: MediaPipe FaceMesh.
- **Voice Recognition**: FasterWhisper for optimized speech-to-text conversion.

### Dataset
- **Gesture Dataset**: 
  - Augmented from YouTube frames with diverse conditions.
  - Techniques: brightness adjustment, shearing, zoom.
- **Facial Recognition Models**: Pre-trained models for integration.
- **Voice Recognition**: Predefined word list for comparison.

