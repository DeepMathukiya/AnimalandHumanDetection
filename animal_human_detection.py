import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from pathlib import Path
from tensorflow.keras.models import load_model
import numpy as np
from ultralytics import YOLO

#expand the bounding box to a minimum size
def expand_box_to_min_size(x1, y1, x2, y2, img_width, img_height, min_size=64):
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Ensure width and height are at least `min_size`
    new_w = max(w, min_size)
    new_h = max(h, min_size)

    x1_new = max(cx - new_w // 2, 0)
    y1_new = max(cy - new_h // 2, 0)
    x2_new = min(cx + new_w // 2, img_width)
    y2_new = min(cy + new_h // 2, img_height)

    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)

# Classify the cropped object
def classify_object(model, cropped_img):
    pred = model.predict(cropped_img)  #prediction
    if pred>0.5 :
        return "Human"
    else:
        return "Animal"   


# Process video
def process_video(video_path, output_path, yolo_model, classifier_model):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_h, img_w = frame.shape[:2]    
        results = yolo_model(frame)
        detections = results[0].boxes

        for i, box in enumerate(detections.xyxy.cpu().numpy()): 
            x1, y1, x2, y2 = map(int, box)
            x1_new, y1_new, x2_new, y2_new = expand_box_to_min_size(x1, y1, x2, y2, img_w, img_h)
            cropped = frame[y1_new:y2_new, x1_new:x2_new]
            cropped =cv2.resize(cropped, (64, 64))
            print(cropped.shape)
            img_array = img_to_array(cropped)
            img_array= np.array(img_array)/255.0
            img_array = np.expand_dims(img_array, axis=0)

            if cropped.size == 0:
                continue

            label = classify_object(classifier_model, img_array)

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.imshow("Frame", frame)

        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (img_w, img_h))

        out.write(frame)

    cap.release()
    if out:
        out.release()

# Main function
def main( ):
    yolo = YOLO("models/yolo_model_trained.pt")  # Load YOLO model
    classifier = load_model("models/animal_human_classifier.h5")  # Load classifier model
    video_file = input("Enter the name of the video file to process: ")
    input_dir = "test_videos"
    output_dir = "outputs"
    # output_dir.mkdir(parents=True, exist_ok=True)
    video_path = input_dir + "/"+ video_file
    if os.path.isfile(video_path):
        output_path = output_dir +f"/processed_{video_file}"
        print(f"Processing: {video_file}")
        process_video(str(video_path), str(output_path), yolo, classifier)
        print(f"Saved to: {output_path}")
    else:
        print(f"File {video_file} does not exist in {input_dir}.")    

if __name__ == "__main__":
    main()
