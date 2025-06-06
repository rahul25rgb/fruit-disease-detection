# 🍎 Fruit Ripeness and Disease Detection

A machine learning-based web application for detecting the **ripeness** and **diseases** of fruits using computer vision. Built with Flask and YOLO, this project aims to assist farmers and vendors with real-time fruit analysis.

## 🚀 Features

- 🔍 Detects common fruit diseases from uploaded images
- 🍌 Classifies fruits based on ripeness level (e.g., raw, ripe, overripe)
- 📷 YOLO-based object detection for accurate and fast analysis
- 🌐 Simple web interface built using Flask
- 🐳 Docker-ready setup (optional)
- 🤖 Planned integration with Gemini for interactive explanations (future feature)

## 🛠️ Tech Stack

- Python
- Flask
- OpenCV
- YOLOv8 (via Ultralytics)
- HTML/CSS/Bootstrap (for UI)
- Docker (optional)

## 📦 Installation

```bash
git clone https://github.com/rahul25rgb/fruit-disease-detection.git
cd fruit-disease-detection

# Create a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
