from flask import Flask, request, jsonify
from deepface import DeepFace
import tempfile
import os
import traceback

app = Flask(__name__)

@app.route('/verify', methods=['POST'])
def verify():
    # Check for required files
    if 'img1' not in request.files or 'img2' not in request.files:
        return jsonify({'error': 'Both img1 and img2 must be provided in the request.'}), 400

    img1 = request.files['img1']
    img2 = request.files['img2']

    # Validate file names
    if img1.filename == '' or img2.filename == '':
        return jsonify({'error': 'Empty filename received for one or both images.'}), 400

    try:
        # Save images to temporary files
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
                # Run DeepFace verification
                result = DeepFace.verify(temp1.name, temp2.name, enforce_detection=False)
                return jsonify(result)
            except ValueError as ve:
                return jsonify({'error': 'Invalid image format or face not detected.', 'details': str(ve)}), 422
            except Exception as deepface_error:
                traceback.print_exc()
                return jsonify({
                    'error': 'Face verification failed during DeepFace processing.',
                    'details': str(deepface_error)
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
