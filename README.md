# Object Detection and Classification Pipeline (YOLO + Mobilenetv2)

This repository provides a Python script that:

- Loads videos from a specified input directory
- Detects objects using a YOLO re trained model on Human head and Animal 
- Classifies detected objects using a Mobilenet
- Annotates the video with labels and bounding boxes
- Saves the final processed video to an output directory

---


## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/DeepMathukiya/AnimalandHumanDetection
cd AnimalandHumanDetection
```
### 2.Create a virtual environment (optional)

For Windows
```bash 
python -m venv venv
venv/bin/activate      
```
For Ubuntu
```bash 
python -m venv venv
source venv/bin/activate      
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. How to Run

#### 1. Add video files

Place your .mp4 video files inside the test_videos/ directory.

#### 2. Run the script for save the video

```bash
python animal_human_detection.py
```
#### 3. Run script for Realtime Video URL  
provide Video Link at run time
eg rtmp://localhost:1935/live
```bash 
python animal_human_detection_realtime.py
``` 

