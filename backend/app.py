from flask import Flask, request, jsonify
from deepface import DeepFace

app = Flask(__name__)

@app.route('/verify', methods=['POST'])
def verify():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({'error': 'Both img1 and img2 are required'}), 400

    img1 = request.files['img1']
    img2 = request.files['img2']

    try:
        result = DeepFace.verify(img1.read(), img2.read(), enforce_detection=False)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
