# Skin Disease Classification using Transfer Learning

This project compares three deep learning architectures for automatic skin disease classification:

- ResNet-50
- EfficientNet-B4
- ViT-B/16

The objective is to classify dermatological images into three classes:

- Melanoma
- Carcinoma
- Eczema

## Dataset

The final dataset contains 2,642 images:

- 2,113 training images
- 264 validation images
- 265 test images

The images come mainly from HAM10000 for melanoma and carcinoma, and DermNet NZ for eczema.

The full dataset is not included in this repository because of file size and redistribution limitations.

## Methodology

The project uses Transfer Learning with pretrained ImageNet weights.  
All models are trained using PyTorch, AdamW optimizer, weighted CrossEntropyLoss, and early stopping.

## Results

All three models achieved 98.87% accuracy on the test set.

## Explainability

Grad-CAM is used to visualize the image regions that influence model predictions.

## Interface

A Streamlit interface is provided in:

```bash
app/streamlit_app.py
```

To run the interface:

```bash
streamlit run app/streamlit_app.py
```

## Repository Structure

```text
skin-disease-transfer-learning/
│
├── notebooks/
│   └── Project_DL.ipynb
│
├── app/
│   └── streamlit_app.py
│
├── report/
│   └── Projet_DL_Rapport.pdf
│
├── presentation/
│   └── Présentation.pptx
│
├── images/
│   └── results/
│       ├── gradcam_example.png
│       └── streamlit_interface.png
│
├── results/
│   ├── classification_reports/
│   ├── confusion_matrices/
│   ├── gradcam/
│   ├── learning_curves/
│   └── roc_curves/
│
├── models/
│   └── README.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Medical Warning

This system is an experimental educational prototype.  
It does not replace the diagnosis or advice of a dermatologist.