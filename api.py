import os
from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd

from src.model import get_model
from src.constants import MODEL_PATH, IMAGE_SIZE, CSV_PATH

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

ground_truth_df = pd.read_csv(CSV_PATH, sep=";")
ground_truth_map = dict(zip(ground_truth_df["file"], ground_truth_df["angle"]))

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

@app.route("/")
def home():
    return "Model API is running."

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(image_file).convert("RGB")
        filename = os.path.basename(image_file.filename)
    except Exception as e:
        return jsonify({"error": f"Could not open image: {str(e)}"}), 400

    try:
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            predicted_angle = output.item()

        if filename in ground_truth_map:
            real_angle = ground_truth_map[filename]
            difference = predicted_angle - real_angle
            return jsonify({
                "predicted": predicted_angle,
                "real": real_angle,
                "difference": difference
            })
        else:
            return jsonify({
                "predicted": predicted_angle,
                "warning": f"No ground truth available for {filename}"
            })
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run()
    # app.run(debug=True)

