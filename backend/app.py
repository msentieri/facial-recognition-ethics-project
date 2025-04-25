from flask import Flask, request, jsonify
from deepface import DeepFace
import tempfile

app = Flask(__name__)

@app.route('/verify', methods=['POST'])
def verify():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({'error': 'Both img1 and img2 are required'}), 400

    img1 = request.files['img1']
    img2 = request.files['img2']

    try:
        # Save both images to temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp2:

            temp1.write(img1.read())
            temp2.write(img2.read())
            temp1.flush()
            temp2.flush()

            result = DeepFace.verify(temp1.name, temp2.name, enforce_detection=False)

        return jsonify(result)
    except Exception as e:
        print("DeepFace error:", e)  # Log to console
        return jsonify({'error': 'Failed to analyze image. Please try another.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
