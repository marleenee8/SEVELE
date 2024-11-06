# Before starting do this in cmd:
# pip install flask torch flask-cors numpy

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np

app = Flask(__name__)
CORS(app)

# Just an Example PyTorch model (Will be replaced with actual model)
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()

@app.route('/process-content/', methods=['POST'])
def process_content():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    contents = file.read()
    image = np.frombuffer(contents, np.uint8)
    image_tensor = torch.from_numpy(image).float()

    # Pytorch operations done here (Getting Recommendations)
    recommendation = model(image_tensor)

    result = {"status": "success", "data": recommendation.tolist()}
    return jsonify(result)

@app.route('/get-recommendations/', methods=['POST'])
def get_recommendations():
    user_data = request.json
    # Process user data and generate recommendations
    # This is just a placeholder for the actual recommendation logic
    user_tensor = torch.tensor(user_data['features'], dtype=torch.float32)
    recommendations = model(user_tensor).tolist()

    result = {"status": "success", "recommendations": recommendations}
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
