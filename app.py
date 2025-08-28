# app.py
from pathlib import Path
from io import BytesIO
from PIL import Image
import torch, torchvision
from torchvision import transforms
from flask import Flask, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT   = Path("weights/best_model.pt")

# Load model + label maps ------------------------------------------------
checkpoint = torch.load(CKPT, map_location=DEVICE)
roast_map  = checkpoint["roast_map"];  inv_roast = {v:k for k,v in roast_map.items()}
defect_map = checkpoint["defect_map"]; inv_def   = {v:k for k,v in defect_map.items()}

class MultiHeadNet(torch.nn.Module):
    def __init__(self, n_roast, n_defect):
        super().__init__()
        self.backbone = torchvision.models.mobilenet_v2().features
        self.pool  = torch.nn.AdaptiveAvgPool2d(1)
        dim = 1280
        self.head_bin = torch.nn.Linear(dim, 2)
        self.head_roa = torch.nn.Linear(dim, n_roast)
        self.head_def = torch.nn.Linear(dim, n_defect)
    def forward(self, x):
        f = self.pool(self.backbone(x)).flatten(1)
        return self.head_bin(f), self.head_roa(f), self.head_def(f)

model = MultiHeadNet(len(roast_map), len(defect_map)).to(DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

tfm = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])
])

# Flask ------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # 6 MB limit (accounts for base64/multipart overhead)

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        "error": "Uploaded file is too large",
        "limit_bytes": app.config["MAX_CONTENT_LENGTH"],
        "received_bytes": request.content_length
    }), 413

@app.route("/")
def index():
    return "Coffee bean classifier – API is live."

@app.route("/classify/", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error":"no file"}), 400

    img = Image.open(BytesIO(request.files["file"].read())).convert("RGB")
    x   = tfm(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out_bin, out_roa, out_def = model(x)
        p_bin  = torch.softmax(out_bin,1).cpu()[0]      # size 2
        p_roa  = torch.softmax(out_roa,1).cpu()[0]      # n_roast
        p_def  = torch.softmax(out_def,1).cpu()[0]      # n_def

    result = {
        "normal": float(p_bin[0]),
        "defect": float(p_bin[1]),
        "probabilities": {
            "roast": {inv_roast[i]: float(p_roa[i]) for i in range(len(inv_roast))},
            "defect": {inv_def[i]: float(p_def[i]) for i in range(len(inv_def))}
        }
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)