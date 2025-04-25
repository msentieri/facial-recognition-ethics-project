from flask import Flask, request, jsonify
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import tempfile
import traceback

from facenet_pytorch import MTCNN
from ghostfacenet.model import GhostFaceNet

app = Flask(__name__)

# Load face detector and model once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=112, margin=0, device=device)
model = GhostFaceNet(embedding_size=512)
model.load_state_dict(torch.load("ghostfacenet.pth", map_location=device))
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)
    if face is None:
        return None
    face = transform(face).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(face)
    return embedding.cpu().numpy()

def cosine_similarity(a, b):
    return float(np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.route('/verify', methods=['POST'])
def verify():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({'error': 'Both img1 and img2 must be provided.'}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f2:
            f1.write(request.files['img1'].read())
            f2.write(request.files['img2'].read())
            f1_path, f2_path = f1.name, f2.name

        emb1 = get_embedding(f1_path)
        emb2 = get_embedding(f2_path)

        os.unlink(f1_path)
        os.unlink(f2_path)

        if emb1 is None or emb2 is None:
            return jsonify({'error': 'Face not detected in one or both images.'}), 422

        similarity = cosine_similarity(emb1, emb2)
        return jsonify({
            'verified': similarity > 0.6,  # You can tune this threshold
            'similarity': similarity
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Verification failed', 'details': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    app.run(debug=True)
