from flask import Flask, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app) 

MXFACE_API_KEY = os.getenv('MXFACE_API_KEY')

@app.route('/api-key', methods=['GET'])
def get_api_key():
    if not MXFACE_API_KEY:
        return jsonify({'error': 'API key not configured'}), 500
    
    return jsonify({
        'api_key': MXFACE_API_KEY,
        'api_url': 'https://faceapi.mxface.ai/api/v3/face/verify'
    })

if __name__ == "__main__":
    app.run()
