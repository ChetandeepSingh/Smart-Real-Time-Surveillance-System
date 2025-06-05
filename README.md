# 🔐 Smart Real-Time Surveillance System

A modular, real-time surveillance system that detects **face mask violations**, **weapons (guns & knives)**, and **violent activities** using deep learning and computer vision.

Built with **YOLOv8n**, **MobileNetV2**, and **OpenAI CLIP**, the system processes live video feeds, triggers hardware-level alerts, and sends automatic email notifications when suspicious behavior is detected.

---

## 🚀 Features

- 🧠 **Mask Detection** using MobileNetV2 (Keras/TensorFlow)
- 🔫 **Weapon Detection (Gun & Knife)** using YOLOv8n (Ultralytics, PyTorch)
- ⚔️ **Violence Detection** using OpenAI CLIP (ViT-B/32) with zero-shot classification
- 📩 **Email Alerts** with attached video evidence (SMTP)
- 🔊 **Hardware Alerts** via Serial Communication (buzzer, LEDs, etc.)
- 🎥 **Automatic Recording** of threat events (AVI format)

---

## 🛠️ Tech Stack

- `Python`
- `OpenCV`
- `YOLOv8n` (Ultralytics)
- `TensorFlow / Keras` (MobileNetV2)
- `PyTorch` (CLIP)
- `SMTP` (for Email)
- `pySerial` (for hardware communication)

---
