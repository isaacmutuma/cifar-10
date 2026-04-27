
# CIFAR-10 Image Classifier — Convolutional Neural Network

A convolutional neural network (CNN) built from scratch in PyTorch to classify images from the CIFAR-10 dataset across 10 categories. This project is part of a self-directed machine learning curriculum building toward research in computer vision and robotics.

## Results

| Epoch | Accuracy |
|-------|----------|
| 1     | 59.4%    |
| 2     | 65.9%    |
| 3     | 69.3%    |
| 4     | 70.7%    |
| 5     | 71.2%    |
| 6     | 72.0%    |
| 7     | 71.8%    |
| 8     | 71.7%    |
| 9     | 71.8%    |
| 10    | 71.8%    |

**Final accuracy: 71.8% on the CIFAR-10 test set** (10,000 images, 10 classes)

Loss dropped from 2.31 in epoch 1 to under 0.20 by epoch 10, confirming stable convergence.

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60,000 colour images across 10 classes:

```
0: Airplane    5: Dog
1: Automobile  6: Frog
2: Bird        7: Horse
3: Cat         8: Ship
4: Deer        9: Truck
```

- 50,000 training images
- 10,000 test images
- Image shape: (3, 32, 32) — RGB, 32×32 pixels

## Model Architecture

```
Input: (3, 32, 32) — RGB image

Conv Block 1:
  Conv2d(3 → 32, kernel=3, padding=1)
  ReLU
  MaxPool2d(2) → output: (32, 16, 16)

Conv Block 2:
  Conv2d(32 → 64, kernel=3, padding=1)
  ReLU
  MaxPool2d(2) → output: (64, 8, 8)

Classifier:
  Flatten → (4096,)
  Linear(4096 → 512)
  ReLU
  Linear(512 → 10)

Output: 10 class scores (logits)
```

**Total parameters:** ~2.3 million learnable weights

## Training Setup

```python
Loss function:  nn.CrossEntropyLoss()
Optimizer:      Adam (lr=1e-3)
Batch size:     64
Epochs:         10
Device:         CUDA (GPU)
```

## How to Run

Open the notebook in Google Colab:

1. Upload `cifar10_cnn.ipynb` to Google Colab
2. Set runtime to GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells — CIFAR-10 downloads automatically
4. Training takes approximately 10-15 minutes on GPU

## Next Steps

- Add data augmentation (RandomHorizontalFlip, RandomCrop) to push accuracy toward 80%+
- Add Batch Normalisation for more stable training
- Experiment with deeper architectures (ResNet-style skip connections)
- Reproduce ResNet-18 from scratch

## Context

This project is part of a self-directed ML learning path at Pusan National University, building toward research in computer vision and robotic perception. Prior projects include a NumPy-based air quality sensor analysis and PyTorch tensor fundamentals.

**Learning path:** Tensors → Datasets → Transforms → Neural Networks → CNN (this project) → ResNet reproduction → ViT implementation

## Author

Isaac Mutuma — Year 2 CSE, Pusan National University, South Korea
GKS Scholar | github.com/isaacmutuma
