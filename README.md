<a name="readme-top"></a>

<br />
<div align="center">
  <a href="https://github.com/KemalCaan/Graduation-Project">
    <img src="https://cdn-icons-png.flaticon.com/512/3033/3033143.png" alt="Logo" width="100" height="100">
  </a>

  <h3 align="center">Smart Door Lock System Using Face Recognition</h3>

  <p align="center">
    A robust, edge-computing biometric security system built on Raspberry Pi 4 with Anti-Spoofing capabilities!
    <br />
    <a href="https://github.com/KemalCaan/Graduation-Project"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="#demo">View Demo</a>
    ·
    <a href="https://github.com/KemalCaan/Graduation-Project/issues">Report Bug</a>
    ·
    <a href="https://github.com/KemalCaan/Graduation-Project/issues">Request Feature</a>
  </p>
</div>

<div align="center">
  
[![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow_Lite-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry_Pi_4-C51A4A?style=for-the-badge&logo=raspberry-pi&logoColor=white)](https://www.raspberrypi.org/)

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
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

---

## 🧐 About The Project

<div align="center">
  <img src="https://via.placeholder.com/800x400.png?text=Place+Your+Project+Demo+GIF/Image+Here" alt="Project Demo">
</div>
<br/>

This project is a fully offline, edge-computing based **Biometric Smart Door Lock System**. Unlike traditional systems, it eliminates cloud dependency and ensures high-level security against **spoofing attacks** (photos/videos) using a custom Liveness Detection model.

It detects faces using **MTCNN**, extracts facial embeddings via **InceptionResNetV1 (FaceNet)**, and prevents presentation attacks using **TensorFlow Lite**.

### 🌟 Key Highlights:
* **Anti-Spoofing (Liveness Detection):** Prevents unauthorized access via photos or videos with **98% accuracy**.
* **Privacy First (KVKK/GDPR Compliant):** All biometric data is processed and stored locally on the Raspberry Pi.
* **Custom Hardware Driver:** A purpose-built **MOSFET (BS170) circuit** safely drives the 5V solenoid lock directly from the 3.3V GPIO pins.
* **One-Shot Registration:** Instantly register a new user without model retraining.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### 🛠 Built With

* [![Python][Python-badge]][Python-url]
* [![OpenCV][OpenCV-badge]][OpenCV-url]
* [![PyTorch][PyTorch-badge]][PyTorch-url]
* [![TensorFlow][TF-badge]][TF-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🏗 System Architecture

The software pipeline is highly optimized for the limited resources of a Raspberry Pi 4, executing in three concurrent threads to maintain a responsive Tkinter GUI.

1. **Preprocessing & Alignment:** MTCNN accurately locates the face and 5 landmarks (eyes, nose, mouth corners) even under varying lighting.
2. **Liveness Check:** The cropped face is passed to a TFLite Anti-Spoofing model. If the spoof score is < 0.5, access is immediately denied.
3. **Embedding Extraction & Matching:** A 512-dimensional vector is generated using InceptionResNetV1 and compared via Cosine Similarity (Threshold: > 0.90).

<div align="center">
  <img src="https://via.placeholder.com/600x400.png?text=Add+Figure+8+(Workflow+Diagram)+Here" alt="Workflow">
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

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

<div align="center">
  <img src="https://via.placeholder.com/600x300.png?text=Add+Figure+2+(Hardware+Design)+Here" alt="Hardware Setup">
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 📊 Performance & Results

We rigorously tested our system against traditional methods across 1,000+ positive and negative validation frames.

* **Overall Accuracy:** **96.5%** (Significantly outperforms Haar Cascade+HOG and VGG16+SSD300).
* **Liveness Validation:** Successfully rejected 96% of all spoofing attempts (paper photos, phone screens).
* **Speed:** Maintains a stable **~0.7 FPS** end-to-end processing speed on embedded hardware without external GPU accelerators.
* **Low Light Durability:** Retains 92.5% accuracy even under 100-200 lux conditions.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🚀 Getting Started

Follow these steps to set up the project locally on your Raspberry Pi.

### Prerequisites
* Raspberry Pi OS (64-bit recommended)
* Python 3.8 or higher

### Installation
1. Clone the repo
   ```sh
   git clone [https://github.com/KemalCaan/Graduation-Project.git](https://github.com/KemalCaan/Graduation-Project.git)
