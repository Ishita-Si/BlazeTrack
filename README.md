 ## BlazeTrack : Forest Fire Detection System

Detecting fire vs. non-fire from images using deep learning
*Built with TensorFlow, Streamlit & Kaggle datasets*

![Model](https://img.shields.io/badge/Model-CNN%20%7C%20MobileNetV2-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-Up%20to%2095%25-brightgreen)
![Dataset](https://img.shields.io/badge/Data-Kaggle%3A%20phylake1337%2Ffire--dataset-orange)
![Status](https://img.shields.io/badge/Status-Under%20Development-yellow)

---

### Dataset

Dataset used: [phylake1337/fire-dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset)

* Contains two folders:

  * `fire_images/`: Images with visible fire.
  * `non-fire_images/`: Images without fire.
* Total: \~1,000 images (approx. 755 fire, 250 non-fire)
</br> will continue to update with better datasets

---

### Models

1. **Custom CNN (from scratch)**

   * Built using basic `Conv2D`, `MaxPooling`, `Dropout`, `Dense`.
   * Input size: 128x128x3
   * Accuracy: \~87% (can vary)

2. **Transfer Learning (MobileNetV2)**

   * Pretrained weights from ImageNet.
   * Fine-tuned on our fire dataset.
   * Accuracy: \~92% on validation set.

---

### EDA

* Image distribution
* Class imbalance check
* Sample visualization (matplotlib/seaborn)
* Augmentation used for balancing

---

### Streamlit App

* Upload an image to predict `FIRE` or `NON-FIRE`
* Real-time classification with confidence score
* Built using `Streamlit`
* Example usage:

```bash
streamlit run app.py
```

### Installation

```bash
git clone https://github.com/Ishita-Si/BlazeTrack
cd fire-detection
pip install -r requirements.txt
```

---

### Future Improvements

* Add Grad-CAM heatmaps
* Improve non-fire class accuracy
* Use YOLO for real-time video stream detection

