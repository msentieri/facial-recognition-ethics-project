from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

@app.route('/api/facial-recognition', methods=['POST'])
def recognize():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join('/tmp', filename)
    file.save(file_path)
    
    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)

    return jsonify({'faces_detected': len(face_locations)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
