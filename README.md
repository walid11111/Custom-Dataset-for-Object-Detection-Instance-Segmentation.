# Custom Human Detection – Object Detection & Instance Segmentation with YOLOv11
# 📌 Overview
This project implements object detection and instance segmentation on a custom human detection dataset using the YOLOv11 model from Ultralytics.
The model is trained, validated, and tested on a labeled dataset to accurately identify humans in images while also segmenting their shapes.

 #✨ Features
Custom dataset integration for human detection.

YOLOv11n segmentation architecture for high performance.

Hyperparameter tuning for improved accuracy.

Training, validation, and inference workflows.

Visual output for both bounding boxes and segmentation masks.

# 📂 Dataset
The dataset is stored in YOLO format (images, labels, and data.yaml).

It contains segmentation labels for human objects.

Example dataset path:

bash
Copy
Edit
/content/khan/data.yaml
⚙️ Installation
bash
Copy
Edit
# Clone this repository
git clone https://github.com/yourusername/custom-human-detection.git
cd custom-human-detection

# Install dependencies
pip install ultralytics opencv-python numpy
🚀 Training the Model
python
Copy
Edit
from ultralytics import YOLO

# Load pretrained YOLOv11n segmentation model
model = YOLO("yolo11n-seg.pt")

# Train with custom dataset
model.train(
    data="/path/to/data.yaml",
    epochs=40,
    imgsz=640,
    batch=16,
    name="human_detection_segmentation"
)
📊 Validation
python
Copy
Edit
model.val()
🔍 Inference
python
Copy
Edit
results = model.predict(source="/path/to/test/image.jpg", conf=0.25, save=True)
📈 Results
mAP50: <insert value after training>

mAP50-95: <insert value after training>

Example output:
(Insert sample image with bounding boxes and segmentation masks here)

📜 License
This project is licensed under the MIT License.

🙏 Acknowledgements
Ultralytics YOLO

Google Colab for providing GPU runtime

