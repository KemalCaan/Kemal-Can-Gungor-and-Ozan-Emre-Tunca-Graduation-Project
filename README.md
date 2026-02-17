<br />
<div align="center">
  <a href="https://github.com/KemalCaan/Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project">
    <img src="https://cdn-icons-png.flaticon.com/512/2400/2400732.png" alt="Logo" width="100" height="100">
  </a>

  <h3 align="center">Smart Door Lock System Using Face Recognition</h3>

  <p align="center">
    A robust, edge-computing biometric security system built on Raspberry Pi 4 with Anti-Spoofing capabilities!
    <br />
    <br />
    <a href="https://github.com/KemalCaan/Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project/issues">Report Bug</a>
    ·
    <a href="https://github.com/KemalCaan/Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project/issues">Request Feature</a>
  </p>
</div>

<div align="center">
  
[![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow_Lite-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi_4-C51A4A?style=for-the-badge&logo=raspberry-pi&logoColor=white)](https://www.raspberrypi.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

<details>
  <summary>📝 Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#system-architecture">System Architecture</a></li>
    <li><a href="#hardware-setup">Hardware Setup</a></li>
    <li><a href="#performance--results">Performance & Results</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

---

## 🧐 About The Project

This project is a fully offline, edge-computing based **Biometric Smart Door Lock System**. Unlike traditional systems, it eliminates cloud dependency and ensures high-level security against **spoofing attacks** (photos/videos) using a custom **Liveness Detection** model.

It detects faces using **MTCNN**, extracts facial embeddings via **InceptionResNetV1 (FaceNet)**, and prevents presentation attacks using **TensorFlow Lite**.

### 🌟 Key Highlights
* **Anti-Spoofing:** Prevents unauthorized access via photos or videos with **98% accuracy**.
* **Privacy First:** All biometric data is processed and stored locally on the Raspberry Pi (Edge Computing).
* **Custom Hardware Driver:** A purpose-built **MOSFET (BS170) circuit** safely drives the 5V solenoid lock directly from the 3.3V GPIO pins.
* **One-Shot Registration:** Instantly register a new user without model retraining.

### 🛠 Built With
* **Language:** Python 3.9
* **Frameworks:** PyTorch (facenet-pytorch), TensorFlow Lite, OpenCV
* **GUI:** Tkinter
* **Hardware:** Raspberry Pi 4, Pi Camera V3

---

## 🏗 System Architecture

The software pipeline executes in three concurrent threads to maintain a responsive GUI.

1. **Preprocessing & Alignment:** MTCNN accurately locates the face and 5 landmarks.
2. **Liveness Check:** The cropped face is passed to a TFLite Anti-Spoofing model. (Spoof score < 0.5 = Access Denied).
3. **Embedding Extraction:** A 512-dimensional vector is generated using InceptionResNetV1 and compared via Cosine Similarity (Threshold: > 0.90).

> **Note:** Please refer to `Figure 8` in the project documentation for the detailed workflow diagram.
---

## 🔌 Hardware Setup

To bridge the logic gap between the 3.3V logic of the Raspberry Pi and the 5V requirements of the relay/solenoid, a custom hardware interface was designed.

| Component | Function / Spec |
| :--- | :--- |
| **Raspberry Pi 4 Model B** | 4GB RAM, Quad-core Cortex-A72 |
| **Pi Camera Module V3** | Sony IMX219, connected via CSI port |
| **Solenoid Door Lock** | 12V Electromechanical Lock |
| **Relay Module** | 5V Single Channel |
| **Logic Level Converter** | Custom designed using BS170 MOSFET |

---

## 📊 Performance & Results

We rigorously tested our system against traditional methods across 1,000+ positive and negative validation frames.

| Method | Accuracy | Avg. FPS (on RPi4) |
| :--- | :--- | :--- |
| Haar Cascade + HOG | 88.0% | ~3.5 FPS |
| VGG16 + SSD300 | 92.0% | ~1.0 FPS |
| **Our System (MTCNN + FaceNet)** | **96.5%** | **~0.7 FPS** |

* **Liveness Validation:** Successfully rejected 96% of all spoofing attempts.
* **Low Light:** Retains 92.5% accuracy even under 100-200 lux conditions.

---

## 🚀 Installation

1. **Clone the repo**
   ```sh
   git clone [https://github.com/KemalCaan/Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project.git](https://github.com/KemalCaan/Kemal-Can-Gungor-and-Ozan-Emre-Tunca-Graduation-Project.git)
