# Object Detection Learning Timeline (PyTorch Focused)

This roadmap outlines the evolution of **object detection** architectures — from sliding windows and region-based detectors to transformer-based models — highlighting which to **read**, **partially implement**, or **fully implement** for a practical PyTorch learning path.

---

## 1️⃣ Early & Foundational Detectors

### 🧠 **R-CNN (2014)** – ⚙️📖 *Partial Implementation*

* **Key Innovation**: Combined selective search (region proposals) with CNN feature extraction and SVM classification.
* **Learning Focus**: Understand region proposals + CNN-based classification pipeline.
* **Paper**: [Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation](https://arxiv.org/abs/1311.2524)

---

### 🧠 **Fast R-CNN (2015)** – ⚙️ *Must Implement*

* **Key Innovation**: Shared CNN feature maps + ROI pooling for faster detection.
* **Learning Focus**: Implement ROI pooling and shared backbone.
* **Paper**: [Fast R-CNN](https://arxiv.org/abs/1504.08083)

---

### 🧠 **Faster R-CNN (2015)** – ⚙️ *Must Implement*

* **Key Innovation**: Introduced the Region Proposal Network (RPN).
* **Learning Focus**: Implement RPN, anchors, and end-to-end training.
* **Paper**: [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

---

## 2️⃣ Single-Stage Detectors (Speed-Oriented)

### 🧠 **YOLO v1 (2016)** – ⚙️ *Must Implement*

* **Key Innovation**: Replaced proposals with direct bounding box regression.
* **Learning Focus**: Grid-based detection and bounding box parameterization.
* **Paper**: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

---

### 🧠 **SSD (2016)** – ⚙️📖 *Partial Implementation*

* **Key Innovation**: Multi-scale feature maps for detecting different object sizes.
* **Learning Focus**: Feature pyramid concept and anchor-based detection.
* **Paper**: [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)

---

### 🧠 **YOLOv2 / YOLO9000 (2017)** – 📖 *Read Only*

* **Key Innovation**: Introduced batch norm, anchor boxes, and multi-scale training.
* **Learning Focus**: Learn improvements over YOLOv1.
* **Paper**: [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)

---

### 🧠 **RetinaNet (2017)** – ⚙️📖 *Partial Implementation*

* **Key Innovation**: Introduced **Focal Loss** to handle class imbalance.
* **Learning Focus**: Implement Focal Loss and understand one-stage vs two-stage trade-offs.
* **Paper**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

---

## 3️⃣ Modern Object Detectors

### 🧠 **YOLOv3 (2018)** – ⚙️📖 *Partial Implementation*

* **Key Innovation**: Multi-scale prediction, residual backbones (Darknet-53).
* **Learning Focus**: Explore detection across different scales and feature maps.
* **Paper**: [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)

---

### 🧠 **YOLOv5 (2020)** – ⚙️ *Implement Using Codebase*

* **Key Innovation**: PyTorch-native, modern training pipelines, anchors, augmentation.
* **Learning Focus**: Use pretrained models, fine-tune, and visualize training.
* **Repo**: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

---

### 🧠 **EfficientDet (2020)** – ⚙️📖 *Partial Implementation*

* **Key Innovation**: Compound scaling + BiFPN for efficient feature fusion.
* **Learning Focus**: Implement BiFPN block and scaling rules.
* **Paper**: [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)

---

### 🧠 **DETR (2020)** – ⚙️ *Must Implement (Simplified)*

* **Key Innovation**: Transformer-based detection removing need for anchors/NMS.
* **Learning Focus**: Implement a minimal DETR with transformer encoder-decoder.
* **Paper**: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

---

### 🧠 **YOLOv8 (2023)** – 📖 *Read Only*

* **Key Innovation**: Anchor-free design + hybrid backbone.
* **Learning Focus**: Understand current state-of-the-art anchor-free trends.
* **Repo**: [ultralytics/YOLOv8](https://github.com/ultralytics/ultralytics)

---

## 🧭 Implementation Path (PyTorch Order)

| Phase                      | Focus                           | What to Implement         | Key Concepts                  |
| -------------------------- | ------------------------------- | ------------------------- | ----------------------------- |
| **1️⃣ Early Two-Stage**    | Learn proposal-based detection  | Fast R-CNN → Faster R-CNN | ROI pooling, RPN, anchors     |
| **2️⃣ One-Stage Models**   | Learn speed-optimized detection | YOLOv1 → RetinaNet        | Anchor boxes, Focal Loss      |
| **3️⃣ Transformer Models** | Learn end-to-end detection      | DETR                      | Attention, NMS-free detection |
| **4️⃣ Modern Practice**    | Apply pretrained models         | YOLOv5 / YOLOv8           | Training, deployment, metrics |

---

## ✅ Summary

* **Must Implement**: Fast R-CNN, Faster R-CNN, YOLOv1, DETR
* **Partial Implementation**: SSD, RetinaNet, YOLOv3, EfficientDet
* **Read Only**: YOLOv2, YOLOv8

