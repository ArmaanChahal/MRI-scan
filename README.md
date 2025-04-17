
# NAVI – NeuroVision AI

This project is an AI-powered MRI tumor classification system that uses a deep learning ResNet50 model to identify tumor types from MRI scans. The model integrates into a web-based front-end for real-time predictions.

## 1. Demo
A sample interface that allows users to upload MRI images and receive classification results in real-time.

---

## 2. Installation
This project was tested on CSIL machines and AWS EC2 Ubuntu 20.04 instances.

### Requirements
- Python 3.10
- Git
- Virtual environment tools

### Setup Instructions
```bash
git clone https://github.com/sfu-cmpt340/2025_1_project_07.git
cd 2025_1_project_07
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dependencies (in `requirements.txt`)
- numpy==1.23.5
- tensorflow-macos==2.10.0
- boto3==1.26.10
- scikit-learn==1.2.2
- matplotlib==3.6.2
- flask
- pillow
- datetime
- werkzeug
- reportlab

### ⚠️ Model File
Due to GitHub's file size limitations, the trained model (`mri_model.keras`) is stored **locally** and is **not published online**. You can request access or obtain it through direct transfer.

Place the model in the `Web/` folder before launching the app.

---

## 3. Reproducing the Project

### Uploading Data to S3
```bash
aws s3 cp "/path/to/local/Training" s3://your-bucket-name/merged_output/Training/ --recursive
aws s3 cp "/path/to/local/Testing" s3://your-bucket-name/merged_output/Testing/ --recursive
```

### EC2 Setup
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv awscli git
```

```bash
git clone https://github.com/sfu-cmpt340/2025_1_project_07.git
cd 2025_1_project_07
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run the web app:
```bash
python3 app.py
```

### Training on EC2
```bash
python mri_training.py
ls -lh mri_model.keras
```

### Downloading from S3 (if model is uploaded)
```bash
aws s3 cp s3://your-bucket/model_output/mri_model.keras .
```

---

## Evaluation
To generate classification reports:
```bash
python evaluate.py --epochs=10 --data=path/to/data
```

---

##  Project Structure
```
2025_1_project_07/
├── Web/               # Frontend Flask app
├── mri_training.py    # Training script
├── evaluate.py        # Evaluation script
├── utils/             # Helper functions
├── data/              # MRI data folders
└── mri_model.keras    # Trained model (stored locally)
```


---

