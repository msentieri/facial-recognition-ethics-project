from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

MXFACE_API_URL = os.getenv('MXFACE_API_URL', 'https://faceapi.mxface.ai/api/v3/face/verify')
MXFACE_API_KEY = os.getenv('MXFACE_API_KEY')

@app.route('/verify', methods=['POST'])
def verify():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({'error': 'Both img1 and img2 must be provided.'}), 400

    img1 = request.files['img1']
    img2 = request.files['img2']

    try:
        encoded_img1 = base64.b64encode(img1.read()).decode('utf-8')
        encoded_img2 = base64.b64encode(img2.read()).decode('utf-8')
        
        payload = {
            "encoded_image1": encoded_img1,
            "encoded_image2": encoded_img2
        }

        headers = {
            "Content-Type": "application/json",
            "subscriptionkey": MXFACE_API_KEY
        }

        mxface_response = requests.post(MXFACE_API_URL, headers=headers, json=payload, timeout=15)

        if mxface_response.ok:
            return jsonify(mxface_response.json())
        else:
            return jsonify({'error': 'MXFace API error', 'details': mxface_response.text}), mxface_response.status_code

    except Exception as e:
        return jsonify({'error': 'Unexpected server error.', 'details': str(e)}), 500

if __name__ == "__main__":
    app.run()
