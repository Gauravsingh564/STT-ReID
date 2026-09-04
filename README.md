# Video-Based Re-Identification Framework

A modular and extensible **PyTorch framework for video-based Person Re-Identification (Video Re-ID)**. The framework combines a **Swin Transformer backbone**, **temporal sequence modeling**, **Temporal Attention**, and **ArcFace classification** to learn robust sequence-level identity representations.

The project is designed for **tracklet-based video datasets** and can be adapted to a wide range of video Re-ID benchmarks.

---

## ✨ Features

- **Swin Transformer backbone** for powerful spatial feature extraction
- **Transformer Encoder** for temporal sequence modeling
- **Temporal Attention** for learning informative frames
- **GeM Pooling** for generalized feature aggregation
- **ArcFace classifier** for discriminative identity learning
- **Soft Triplet Loss** for metric learning
- **CMC and mAP evaluation metrics**
- **k-reciprocal re-ranking** for improved retrieval performance
- Modular architecture for easier experimentation and customization
- Support for generic **tracklet-based video Re-ID datasets**

---

## 📁 Repository Structure

```text
.
├── models/
│   ├── architecture.py
│   ├── layers.py
│   └── loss.py
│
├── utils/
│   └── evaluation.py
│
└── README.md
```

### `models/architecture.py`

Contains the main `VideoReIDModel` architecture.

The model integrates:

- Swin Transformer backbone
- Temporal Transformer Encoder
- Temporal Attention
- Sequence-level feature aggregation
- ArcFace classification

The architecture is designed to transform a sequence of frames into a robust identity representation suitable for both classification and retrieval.

### `models/layers.py`

Contains custom neural network layers and modules:

- **Generalized Mean (GeM) Pooling**
- **TemporalAttention**
- **ArcFace**

These components provide flexible feature aggregation, temporal weighting, and margin-based identity classification.

### `models/loss.py`

Contains the custom **SoftTripletLoss** implementation.

The loss uses soft-margin metric learning to encourage:

- Smaller distances between samples of the same identity
- Larger distances between samples of different identities

An advanced Euclidean distance clamping mechanism is used to improve numerical stability during training.

### `utils/evaluation.py`

Provides utilities for evaluating Video Re-ID performance.

Supported evaluation functionality includes:

- **Cumulative Matching Characteristic (CMC)**
- **mean Average Precision (mAP)**
- **k-reciprocal re-ranking**

Re-ranking can refine query-gallery distances and potentially improve retrieval performance.

---

## 🧠 Model Architecture

The overall processing pipeline is:

```text
Input Video Tracklet
        │
        ▼
Sampled Frames
        │
        ▼
Swin Transformer Backbone
        │
        ▼
Frame-Level Features
        │
        ▼
Temporal Transformer Encoder
        │
        ▼
Temporal Attention
        │
        ▼
Sequence-Level Feature Representation
        │
        ├──────────────────► Retrieval / Re-ID Evaluation
        │
        ▼
ArcFace Classifier
        │
        ▼
Identity Prediction
```

### Processing Flow

1. A video tracklet is represented as a sequence of frames.
2. Each frame is processed by the **Swin Transformer backbone**.
3. Frame-level representations are passed to a **Transformer Encoder** for temporal modeling.
4. **Temporal Attention** assigns importance to different frames.
5. The resulting sequence-level embedding is used for:
   - Identity classification through **ArcFace**
   - Metric learning through **SoftTripletLoss**
   - Retrieval and Re-ID evaluation

---

## 📊 Supported Datasets

The framework is generic and can process any appropriately prepared **tracklet-based video sequence dataset**.

The following datasets are commonly used for benchmarking.

### 1. PRID-2011

PRID-2011 contains video sequences captured using two static surveillance cameras.

**Highlights:**

- 200 identities
- 400 image sequences
- Two camera views
- Significant background and illumination differences
- Viewpoint variation
- Multiple poses per identity

Dataset page:

https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/prid11

---

### 2. MARS

MARS (**Motion Analysis and Re-identification Set**) is a large-scale video-based person Re-ID dataset and a video extension related to Market-1501.

**Highlights:**

- 1,261 identities
- Approximately 20,000 tracklets
- Automatically generated tracklets
- DPM pedestrian detection
- GMMCP tracking

Dataset page:

https://opendatalab.com/OpenDataLab/MARS

---

### 3. iLIDS-VID

iLIDS-VID contains video sequences for 300 identities captured by two non-overlapping cameras.

**Highlights:**

- 300 identities
- Two sequences per identity
- Two non-overlapping cameras
- Challenging surveillance scenarios

Dataset download:

https://xiatian-zhu.github.io/downloads_qmul_iLIDS-VID_ReID_dataset.html

---

## ⚙️ Installation

Create and activate a Python environment:

```bash
python -m venv venv
```

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install PyTorch and the project dependencies:

```bash
pip install torch torchvision
```

Install any additional dependencies required by your implementation.

---

## 🚀 Usage

The framework is intended to be integrated into a standard Video Re-ID training pipeline.

A typical workflow is:

```text
1. Prepare the dataset
2. Create train/query/gallery tracklets
3. Sample frames from each tracklet
4. Initialize VideoReIDModel
5. Train with classification and metric-learning objectives
6. Extract sequence-level embeddings
7. Evaluate using CMC and mAP
8. Optionally apply k-reciprocal re-ranking
```

> **Note:** The exact constructor arguments and training loop depend on the implementation in `models/architecture.py`.

---

## 🎯 Training Objectives

The framework combines complementary learning objectives.

### ArcFace Classification

ArcFace introduces an angular margin between identity classes, encouraging more discriminative embeddings.

Benefits include:

- Improved inter-class separation
- Better intra-class compactness
- More discriminative identity features

### Soft Triplet Loss

SoftTripletLoss performs metric learning by encouraging:

- Positive samples to remain close
- Negative samples to remain distant

This improves embedding quality for retrieval-based Re-ID tasks.

---

## 📈 Evaluation

Video Re-ID evaluation typically compares query embeddings against gallery embeddings.

### CMC

The **Cumulative Matching Characteristic (CMC)** measures the probability that the correct identity appears within the top-*k* retrieval results.

Commonly reported ranks include:

- Rank-1
- Rank-5
- Rank-10

### mAP

**mean Average Precision (mAP)** evaluates retrieval quality across all relevant gallery matches.

It is particularly useful when multiple correct matches exist in the gallery.

### k-Reciprocal Re-Ranking

The included re-ranking algorithm refines the initial distance matrix using neighborhood relationships between query and gallery samples.

This can improve retrieval accuracy when the initial feature space contains meaningful local structure.

---

## 🗂️ Dataset Preparation

Datasets should be organized as tracklets, where each tracklet contains multiple frames belonging to the same identity.

A conceptual structure may look like:

```text
dataset/
├── train/
│   ├── 0001/
│   │   ├── tracklet_01/
│   │   │   ├── frame_0001.jpg
│   │   │   ├── frame_0002.jpg
│   │   │   └── ...
│   │   └── tracklet_02/
│   │
│   └── 0002/
│
├── query/
│   └── ...
│
└── gallery/
    └── ...
```

The exact dataset loader can be adapted to the directory structure of the selected benchmark.

---



## 📚 Citation

If you use this repository in academic work, please add appropriate citations for:

- Swin Transformer
- ArcFace
- Triplet-loss-based metric learning
- Any datasets used in your experiments

---


**Built with PyTorch for modular and extensible Video-Based Re-Identification research.**
