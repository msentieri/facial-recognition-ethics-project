import face_recognition
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/facial-recognition', methods=['POST'])
def facial_recognition_api():
    file = request.files['file']
    image = face_recognition.load_image_file(file)
    face_locations = face_recognition.face_locations(image)
    return jsonify({'face_locations': face_locations})

if __name__ == '__main__':
    app.run(debug=True)
