# UpperGI-YOLO

## 1. Classification (`Upper_GI_classification.ipynb`)

| | |
|---|---|
| Model | YOLOv8n-cls |
| Task | Binary classification (`abnormal` / `normal`) |
| Input size | 224 × 224 |
| Training | 30 epochs, batch size 32 |

**Training data:** normal, polyp, ulcer and cancer images combined from Kvasir,
GastroHun, GastroVision and EDD2020.

**External test data:** polyp and normal images from the Gastrointestinal Endoscopy
Polyp dataset and the E-endoscopy BE/RE dataset.

**Evaluation:** accuracy, precision, recall and F1 (abnormal = positive class) on the
validation split, plus separate checks on held-out abnormal and normal sets.

**Held-out test observations**
- Abnormal set: 3 of 42 images misclassified as normal (low confidence)
- Normal set: 1 of 50 images misclassified as abnormal (0.94 confidence)

## 2. Object detection (`UpperGI_object_detection.ipynb`)

| | |
|---|---|
| Model | YOLO11n |
| Dataset | Kvasir (~2,000 images), annotated and exported via Roboflow |
| Classes | `esophagitis`, `polyps`, `ulcerative-colitis` |
| Input size | 640 × 640 |
| Training | 100 epochs, batch size 16 |

**Training strategy:** strong augmentation to reduce overfitting (HSV jitter, rotation,
translation, scaling, flips, mosaic, MixUp) plus dropout and weight decay.

**Evaluation:** a custom `TestSetValidator` computes per-image and per-class precision,
recall and F1 using IoU matching (IoU ≥ 0.5, confidence ≥ 0.25), with distribution plots.

## Getting started

Both notebooks were developed in **Google Colab** with a GPU runtime.

```bash
pip install ultralytics roboflow opencv-python-headless torch torchvision pillow scikit-learn pandas matplotlib pyyaml
```

1. Open a notebook in Colab (or locally in Jupyter).
2. Update the dataset and weight paths (Google Drive paths in the classification notebook,
   `/content/...` paths in the detection notebook).
3. For detection, set your own Roboflow API key (see below) to download the dataset.
4. Run the cells in order: train, validate, then test.

## Configuration notes

- **Never commit API keys.** Load your Roboflow key from an environment variable or Colab
  secrets instead of hardcoding it:
```python
  import os
  rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
```
- Trained weights (`best.pt`) and datasets are not included; add them to `.gitignore`.

## Datasets

- [Kvasir](https://datasets.simula.no/kvasir/)
- GastroHun, GastroVision, EDD2020
- Gastrointestinal Endoscopy Polyp dataset, E-endoscopy BE/RE

Please follow each dataset's license and citation requirements.

## Limitations

- Small datasets and a single-split evaluation; results may not generalize across
  endoscope types, hospitals or patient populations.

## Acknowledgements

Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and Roboflow.

## License

Choose a license (e.g. MIT) and add a `LICENSE` file.
