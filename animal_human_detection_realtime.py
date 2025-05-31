import os
import cv2
import threading
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import validators 

frame = None
running = True
lock = threading.Lock()

# Expand the bounding box to a minimum size
def expand_box_to_min_size(x1, y1, x2, y2, img_width, img_height, min_size=64):
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    new_w = max(w, min_size)
    new_h = max(h, min_size)
    x1_new = max(cx - new_w // 2, 0)
    y1_new = max(cy - new_h // 2, 0)
    x2_new = min(cx + new_w // 2, img_width)
    y2_new = min(cy + new_h // 2, img_height)
    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)

# Classify the cropped object
def classify_object(model, cropped_img):
    pred = model.predict(cropped_img, verbose=0)
    return "Human" if pred > 0.5 else "Animal"

# Thread to read frames from video
def read_frames(cap):
    global frame, running
    while running:
        ret, new_frame = cap.read()
        if not ret:
            running = False
            break
        with lock:
            frame = new_frame

# Thread to process and display frames
def process_frames(yolo_model, classifier_model):
    global frame, running
    while running:
        with lock:
            current_frame = frame.copy() if frame is not None else None
        if current_frame is None:
            continue

        img_h, img_w = current_frame.shape[:2]
        results = yolo_model(current_frame)
        detections = results[0].boxes

        for i, box in enumerate(detections.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            x1_new, y1_new, x2_new, y2_new = expand_box_to_min_size(x1, y1, x2, y2, img_w, img_h)
            cropped = current_frame[y1_new:y2_new, x1_new:x2_new]
            label = int(detections.cls[i].item())
            label_name = "Human" if label == 0 else "Animal"
            print(label_name)

            if cropped.size == 0:
                continue

            cropped = cv2.resize(cropped, (64, 64))
            img_array = img_to_array(cropped) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            label = classify_object(classifier_model, img_array)
            if label != label_name:
                label = "Unknown"  # If classification does not match YOLO label
            

            cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(current_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("Processed Video", current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

# Main function
def main():
    global running
    video_path = input("Enter full path of the video file to process: ")

    if not validators.url(video_path):
        print(f"It is not a valid URL.")
        return       
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    yolo = YOLO("models/yolo_model_trained.pt")
    classifier = load_model("models/animal_human_classifier.h5")

    print("Press 'q' to quit.")
    t1 = threading.Thread(target=read_frames, args=(cap,))
    t2 = threading.Thread(target=process_frames, args=(yolo, classifier))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    cap.release()
    cv2.destroyAllWindows()
    print("Processing finished.")

if __name__ == "__main__":
    main()
