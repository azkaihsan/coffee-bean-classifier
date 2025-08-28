# Coffee-Bean Hierarchical Classifier

A lightweight, end-to-end solution that  
1. Detects whether a coffee-bean sample is **Normal** or **Defect**  
2. If *Normal* → classifies roast colour (green, light, medium, dark)  
3. If *Defect* → identifies the type of defect (full-black, dead, fungus, insect, …)  

The system is designed for on-device inference on an NVIDIA Jetson Nano, whose GPU and CUDA cores drastically shorten training and inference time compared to boards such as the Raspberry Pi [1].

---

## 📌 Quick Links
• Interactive design board on Miro → **https://miro.com/app/board/uXjVP3D27zs=/?moveToWidget=3458764541879399567&cot=14**  
• Jump to: [System Requirements](#system-requirements) · [Model Details](#model-details) · [Docker (optional)](#docker-optional) · [Tips for Production](#tips-for-production) · [License](#license)

---

## 1. Project Highlights
* Hierarchical CNN with three heads  
  – Binary head: Normal / Defect  
  – Roast-colour head (only if Normal)  
  – Defect-type head (only if Defect)  
* Data augmentation: rotation, flips, random crops, colour jitter  
* Metrics: Accuracy, Precision, Recall, F1, ROC-AUC  
* Flask API that returns full probability vectors for each class  
* Ready for TensorRT conversion on Jetson Nano GPU for real-time inference [1]

---

## 2. Folder Structure
```
.
├── dataset/
│   ├── train/
│   │   ├── normal/
│   │   │   ├── green/  …  dark/
│   │   └── defect/
│   │       ├── full_black/ … insect/
│   └── test/              # same structure as train
├── train_eval.py          # training / evaluation script
├── app.py                 # Flask inference server
├── Dockerfile             # (optional) container build
└── weights/               # best_model.pt will be saved here
```

---

## System Requirements
### Hardware
* **NVIDIA Jetson Nano 4 GB** (or any CUDA-capable GPU) – recommended for edge deployment because its CUDA cores accelerate CNN workloads [1]  
* At least 16 GB disk space (dataset + checkpoints)  
* 4 GB RAM minimum

### Software
| Component | Version (tested) |
|-----------|------------------|
| Python    | ≥ 3.8            |
| PyTorch   | ≥ 1.12           |
| Torchvision | ≥ 0.13        |
| scikit-learn | ≥ 1.0        |
| Flask     | ≥ 2.0           |
| Pillow    | ≥ 9             |

Jetson users should install NVIDIA’s pre-compiled PyTorch wheel for aarch64.  

```bash
# Jetson Nano
pip install --extra-index-url https://download.pytorch.org/whl/jp cast torch torchvision
```

---

## Model Details
| Item                | Value |
|---------------------|-------|
| Backbone            | MobileNetV2 (`torchvision.models.mobilenet_v2`) |
| Input size          | 224 × 224 RGB |
| Pre-processing      | Resize 256 → Center/Random crop 224, Normalise ImageNet mean/std |
| Heads (outputs)     | 1) 2-class softmax (Normal / Defect)  <br>2) *n*-class roast softmax <br>3) *m*-class defect softmax |
| Loss Function       | Sum of cross-entropies with masking (only relevant head contributes) |
| Optimiser           | Adam (LR = 3e-4) + CosineAnnealingLR |
| Epochs (default)    | 20 |
| Checkpoint          | Best validation accuracy on binary task (`weights/best_model.pt`) |
| Inference speed     | ≈ 10 FPS on Jetson Nano (batch = 1, FP16 TensorRT) |

---

## Docker (optional)
A container image is useful for reproducibility or cloud deployment.

```dockerfile
# Dockerfile
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir \
        scikit-learn flask pillow

EXPOSE 5000
CMD ["python", "app.py"]
```

Build & run:

```bash
docker build -t coffee-classifier .
docker run -p 5000:5000 coffee-classifier
```

---

## 3. Installation (bare-metal)
```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision scikit-learn flask pillow
```

---

## 4. Training
```bash
python train_eval.py
```
The script prints per-epoch metrics and stores the best checkpoint in `weights/best_model.pt`.

---

## 5. Running the API
```bash
python app.py               # default port 5000
curl -F "file=@bean.jpg" http://localhost:5000/classify/
```
Sample response
```json
{
  "normal": 0.87,
  "defect": 0.13,
  "probabilities": {
    "roast":  { "green":0.02, "light":0.55, "medium":0.28, "dark":0.15 },
    "defect": { "full_black":0.05, "dead":0.03, "fungus":0.02, "insect":0.03 }
  }
}
```

---

## Tips for Production
1. **TensorRT / torch-tensorrt** – convert the `.pt` file to an engine for 2-3× faster inference on Jetson.  
2. **Batch size** – keep batch ≤ 4 on Nano to fit 4 GB memory; use FP16 precision.  
3. **Gunicorn / Uvicorn** – wrap the Flask app with a WSGI server for concurrency.  
4. **Confidence threshold** – return `HTTP 202` *uncertain* if `max(probability) < 0.6`; route to manual inspection.  
5. **Monitoring** – log prediction distributions; drift toward new defect types can be detected by KL divergence.  
6. **Image capture** – use consistent lighting (light-box) and the Razer Kiyo Pro camera at 1080 p/60 fps as in the original setup [1].  
7. **Incremental learning** – append new classes by adding sub-folders, fine-tune for a few epochs (freeze early layers).  
8. **Security** – limit upload size, validate MIME-types, and sandbox model execution if exposing public endpoints.

---

## License
MIT License – see `LICENSE` file for details.

---

## References
[1] A-Coffee-Bean-Classifier-System-by-Roast-Quality … Implemented in an NVIDIA Jetson Nano: the Jetson’s GPU and CUDA cores allow faster training and real-time classification compared to single-board computers like the Raspberry Pi.