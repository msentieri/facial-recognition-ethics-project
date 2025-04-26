from flask import Flask, request, jsonify
import requests
import base64
import os

app = Flask(__name__)

MXFACE_API_URL = os.getenv('MXFACE_API_URL', 'https://faceapi.mxface.ai/api/v3/face/verify')
MXFACE_API_KEY = os.getenv('MXFACE_API_KEY')

def encode_image(file):
    return base64.b64encode(file.read()).decode('utf-8')

def call_mxface(encoded_img1, encoded_img2):
    headers = {
        'Content-Type': 'application/json',
        'subscriptionkey': MXFACE_API_KEY
    }
    payload = {
        "encoded_image1": encoded_img1,
        "encoded_image2": encoded_img2
    }
    response = requests.post(MXFACE_API_URL, headers=headers, json=payload)
    return response

@app.route('/verify', methods=['POST'])
def verify():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({'error': 'Both img1 and img2 must be provided.'}), 400

    img1 = request.files['img1']
    img2 = request.files['img2']

    if img1.filename == '' or img2.filename == '':
        return jsonify({'error': 'Empty filename for one or both images.'}), 400

    try:
        encoded_img1 = encode_image(img1)
        encoded_img2 = encode_image(img2)
        mxface_response = call_mxface(encoded_img1, encoded_img2)

        if mxface_response.status_code == 200:
            return jsonify(mxface_response.json())
        else:
            return jsonify({'error': 'MXFace API error', 'details': mxface_response.text}), mxface_response.status_code

    except Exception as e:
        return jsonify({'error': 'Unexpected server error.', 'details': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
