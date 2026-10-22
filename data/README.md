# Dataset Setup

The dataset used for this project consists of chest X-ray images organized into two classes:

- **HEART** – Chest X-ray images associated with heart abnormalities
- **NORMAL** – Normal chest X-ray images

The image dataset itself is not included in this repository because of its size.

## Expected Directory Structure

Before training or evaluating the model, organize the dataset as follows:

```text
data/
├── train/
│   ├── HEART/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   │
│   └── NORMAL/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│
└── test/
    ├── HEART/
    │   ├── image1.jpg
    │   └── ...
    │
    └── NORMAL/
        ├── image1.jpg
        └── ...
