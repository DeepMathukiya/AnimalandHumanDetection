# Why Open Images Dataset is Ideal for Retraining YOLO Models

The **Open Images Dataset** is a powerful and comprehensive resource curated by Google, containing millions of labeled images spanning thousands of object categories.Apart from this YOLO models are already pre-trained on the Open Images Dataset, using it for fine-tuning or custom classification ensures better compatibility, faster convergence, and more accurate predictions due to the alignment between training and inference data distributions

---

## 🗂 Dataset Summary

| Class Name     | Image Count | Source         |
|----------------|-------------|----------------|
| Animal         | 221         | Open Images v7 |
| human Head     | 225         | Open Images v7 |

- All images have bounding boxes.
- Dataset converted to YOLO format with class labels in `.txt` files.

---
## Dataset For Classification 

To perform the classification task, each image was passed through a YOLO-trained model. Based on the predicted class, the image was cropped and then saved into its corresponding class folder.

