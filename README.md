# DECODE: Dual-Branch EEG Encoding and Cross-Modal Alignment for Visual Neural Decoding

## Table of Contents
- [Introduction](#introduction)
- [Repo Architecture](#repo-architecture)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Run](#run)
- [Acknowledgement](#acknowledgement)

## Introduction
This is the official implementation of **DECODE** (Dual-Branch EEG Encoding and Cross-Modal Alignment for Visual Neural Decoding).

DECODE is a modality-adaptive framework for visual neural decoding from EEG signals that:
- Decouples EEG representation learning into two structurally independent pathways: a lightweight visual branch for EEG–image alignment and a deep spatiotemporal semantic branch for EEG–text alignment
- Introduces a biologically inspired foveated visual prior that mimics the non-uniform cone cell distribution of the human retina
- Employs a cross-modal joint scoring mechanism during inference to fuse complementary visual and semantic retrieval scores

**Keywords:** Visual neural decoding, EEG decoding, cross-modal alignment, dual-branch encoding, foveated visual prior, brain–computer interface

## Repo Architecture
```
DECODE/
├── README.md
├── train_img.py               # Image modality training script
├── train_text.py              # Text modality training script
├── joint_test.py              # Joint image-text testing script
├── base                       # Core implementation files
│   ├── data_eeg.py            # EEG data loading (for joint_test)
│   ├── data_eeg_copy.py       # EEG data loading (for train_img/train_text)
│   ├── eeg_backbone.py        # EEG encoder backbone (visual & semantic branches)
│   ├── inpating_data.py       # Foveated blur preprocessing module
│   └── utils.py               # Utility functions
├── configs
│   └── eeg
│       ├── train_img.yaml     # Configuration for image training
│       ├── train_text.yaml    # Configuration for text training
│       └── joint_test.yaml    # Configuration for joint testing
├── data                       # Directory for datasets (not included)
│   ├── things/                # THINGS image stimuli
│   └── things-eeg/            # Preprocessed EEG data
├── requirements.txt           # List of required Python packages
└── LICENSE
```

## Environment Setup
- Python 3.8+
- CUDA 12.0
- PyTorch 2.4.1
- Required libraries are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Data Preparation
1. Download the THINGS dataset:
   - Things-image from the [OSF repository](https://osf.io/jum2f/files/osfstorage)
   - Things-EEG from the [OSF repository](https://osf.io/anp5v/files/osfstorage) or [Openneuro repository](https://openneuro.org/datasets/ds004212/versions/2.0.1)
   - Place them in the `data` directory

2. Preprocess the EEG data to .pt format

3. Resize the images to the expected input size

The data directory should be structured as:
```
data/
├── things/
│   └── THINGS/
│       ├── Images/
│       └── Metadata/
└── things-eeg/
    ├── Image_set/
    ├── Image_set_Resize/
    └── Preprocessed_data_250Hz_whiten/
```

## Run

### Image modality training
```bash
python train_img.py --config configs/eeg/train_img.yaml \
    --subjects sub-01 --seed 0 \
    --exp_setting intra-subject \
    --brain_backbone EEGProjectLayer \
    --vision_backbone RN50 \
    --epoch 50 --lr 1e-4
```

### Text modality training
```bash
python train_text.py --config configs/eeg/train_text.yaml \
    --subjects sub-01 --seed 0 \
    --exp_setting intra-subject \
    --brain_backbone EEGProjectLayer \
    --vision_backbone RN50 \
    --epoch 50 --lr 1e-4
```

### Joint image-text testing
```bash
python joint_test.py --config configs/eeg/joint_test.yaml \
    --subjects sub-01 \
    --img_checkpoint path/to/img_checkpoint.ckpt \
    --text_checkpoint path/to/text_checkpoint.ckpt \
    --vision_backbone RN50 \
    --text_backbone RN50
```

## Acknowledgement
We acknowledge the contributions of the following datasets:
- [A large and rich EEG dataset for modeling human visual object recognition](https://www.sciencedirect.com/science/article/pii/S1053811922008758) [THINGS-EEG]
- [THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior](https://pubmed.ncbi.nlm.nih.gov/36847339/) [THINGS]

## Citation
If you find our work helpful, please cite:
```bibtex
@article{decode2025,
  title={DECODE: Dual-Branch EEG Encoding and Cross-Modal Alignment for Visual Neural Decoding},
  author={Author One and Author Two and Author Three},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  note={Under review}
}
```

## Contact
For any questions, please contact the authors.
