from flask import Flask, request, jsonify
import face_recognition
import tempfile
import os
import traceback
import numpy as np

app = Flask(__name__)

def compare_faces(img1_path, img2_path):
    # Load images
    image1 = face_recognition.load_image_file(img1_path)
    image2 = face_recognition.load_image_file(img2_path)
    
    # Get face encodings (128-dimension vectors)
    encodings1 = face_recognition.face_encodings(image1)
    encodings2 = face_recognition.face_encodings(image2)
    
    if not encodings1 or not encodings2:
        return {'verified': False, 'reason': 'No faces detected in one or both images'}
    
    # Compare the first face found in each image
    distance = face_recognition.face_distance([encodings1[0]], encodings2[0])[0]
    
    # Convert distance to similarity score (0-1)
    similarity = 1 - distance
    
    # Threshold for verification (adjust as needed)
    verified = similarity > 0.6
    
    return {
        'verified': bool(verified),
        'similarity': float(similarity),
        'distance': float(distance),
        'threshold': 0.6
    }

@app.route('/verify', methods=['POST'])
def verify():
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({'error': 'Both img1 and img2 must be provided in the request.'}), 400

    img1 = request.files['img1']
    img2 = request.files['img2']

    if img1.filename == '' or img2.filename == '':
        return jsonify({'error': 'Empty filename received for one or both images.'}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp2:
            try:
                temp1.write(img1.read())
                temp2.write(img2.read())
                temp1.flush()
                temp2.flush()
            except Exception as file_write_error:
                return jsonify({'error': 'Failed to write image files.', 'details': str(file_write_error)}), 500

            try:
                result = compare_faces(temp1.name, temp2.name)
                return jsonify(result)
            except ValueError as ve:
                return jsonify({'error': 'Invalid image format or face not detected.', 'details': str(ve)}), 422
            except Exception as face_recognition_error:
                traceback.print_exc()
                return jsonify({
                    'error': 'Face verification failed during processing.',
                    'details': str(face_recognition_error)
                }), 500
            finally:
                os.unlink(temp1.name)
                os.unlink(temp2.name)

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'error': 'Unexpected server error occurred.',
            'details': str(e)
        }), 500

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
