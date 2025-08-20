# BaSDNet: Background-Aware Crowd Segmentation and Density Map Generation Network for Kernel-free Crowd Counting

BaSDNet (Background-aware Segmentation guided Density Network) is a deep learning model for crowd counting. It leverages an **ALT-GVT Transformer** backbone and introduces a lightweight **foreground segmentation branch** to suppress background clutter, enabling kernel-free density map regression.

---

## Directory Structure

```
BaSDNet
├── Networks                # Network architectures (ALT-GVT backbone, pipeline, etc.)
├── datasets                # Dataset loading and processing
├── preprocess              # Scripts for preprocessing raw datasets
├── losses                  # OT Loss, Segmentation Loss, etc.
├── utils                   # Logging and utility functions
├── example_images          # Example images
├── train.py                # Training script
├── test_image_patch.py     # Testing/Inference script
├── vis_densityMap.py       # Density map visualization
├── requirements.txt        # Environment dependencies
└── LICENSE.txt             # MIT License
```

---

## Requirements

- Python 3.8+
- PyTorch 1.10.0 + CUDA 11.3
- torchvision 0.11.1
- Other dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

Supported datasets:

- [UCF-QNRF](https://www.crcv.ucf.edu/research/data-sets/ucf-qnrf/)
- [NWPU-Crowd](https://www.crowdbenchmark.com/nwpucrowd.html)
- [ShanghaiTech A/B](https://github.com/desenzhou/ShanghaiTechDataset)
- Custom datasets (must follow the format in `datasets/crowd.py`)

### Preprocessing Example

```bash
python preprocess_dataset.py \
    --dataset qnrf \
    --input-dataset-path /path/to/QNRF \
    --output-dataset-path data/QNRF-Train-Val-Test
```

For NWPU-Crowd, set `--dataset nwpu`.

---

## Training

```bash
python train.py \
    --data-dir <path/to/preprocessed_dataset> \
    --dataset <qnrf|nwpu|sha|shb|custom> \
    --lr 1e-5 \
    --batch-size 10 \
    --device 0
```

Common arguments:

| Argument | Description | Default |
| --- | --- | --- |
| `--max-epoch` | Maximum training epochs | 5000 |
| `--val-epoch` | Validation interval (epochs) | 1 |
| `--wot` | OT Loss weight | 0.1 |
| `--wtv` | TV Loss weight | 0.01 |
| `--wseg` | Segmentation Loss weight | 1.0 |
| `--wandb` | Enable Weights & Biases logging | 0 |

---

## Inference / Testing

```bash
python test_image_patch.py \
    --model-path <path/to/trained_weights> \
    --data-path <dataset_path> \
    --dataset <qnrf|nwpu|sha|shb|custom> \
    --device 0
```

This script performs sliding-window inference and reports the predicted count and error.

---

## Density Map Visualization

```bash
python vis_densityMap.py \
    --image_path <path/to/test_image> \
    --weight_path <path/to/model_weights> \
    --device 0
```

Predicted and ground-truth density maps will be saved to the `vis/` directory.

---

## Benchmark

Performance on the NWPU-Crowd dataset is available on the official leaderboard: [https://www.crowdbenchmark.com/nwpucrowd.html](https://www.crowdbenchmark.com/nwpucrowd.html).

---

## Citation

If you find this project helpful, please cite the repository or related papers:

```text
@misc{BaSDNet,
  title  = {BaSDNet: Background-Aware Crowd Segmentation and Density Map Generation Network for Kernel-free Crowd Counting},
  author = {Repository Contributors},
  year   = {2025},
  note   = {https://github.com/<your-repo>/BaSDNet}
}
```

---

## License

This project is released under the MIT License. See [LICENSE.txt](LICENSE.txt) for details.