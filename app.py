# Before starting do this in cmd:
# pip install flask torch flask-cors numpy PIL io

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Convolutional Neural Network (CNN) for Image Recommendation
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#Simple Model to Calculate User Preference to Liked Content
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # Example output size (10 categories)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Recurrent Neural Network (RNN) for Sequential Data
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

#Transformer Model for Attention-Based Recommendation
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, output_size):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(input_size, num_heads, num_layers)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.transformer(x, x)
        x = self.fc(x[:, -1, :])
        return x


# Initialize models
cnn_model = CNNModel()
model = SimpleNN()
rnn_model = RNNModel(input_size=10, hidden_size=50, output_size=10, num_layers=2)
transformer_model = TransformerModel(input_size=10, num_heads=2, num_layers=2, output_size=10)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.ToTensor(),
])

#Resizing Image to make processing easier
transform = transforms.Compose([
    transforms.Resize((16, 16)),  # Resize to a smaller size for simplicity
    transforms.ToTensor(),
])

@app.route('/process-image', methods=['POST'])
def process_image():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    image = transform(image).unsqueeze(0)

    cnn_model.eval()
    with torch.no_grad():
        output = cnn_model(image)

    recommended_content = {
        'recommended_images': [f'image_{i}.jpg' for i in range(5)],
    }

    return jsonify(recommended_content)

@app.route('/process-sequence', methods=['POST'])
def process_sequence():
    data = request.json.get('sequence')
    sequence = torch.tensor(data).float().unsqueeze(0)

    rnn_model.eval()
    with torch.no_grad():
        output = rnn_model(sequence)

    recommended_content = {
        'recommended_items': [f'item_{i}' for i in range(5)],
    }

    return jsonify(recommended_content)

@app.route('/process-attention', methods=['POST'])
def process_attention():
    data = request.json.get('data')
    attention_data = torch.tensor(data).float().unsqueeze(0)

    transformer_model.eval()
    with torch.no_grad():
        output = transformer_model(attention_data)

    recommended_content = {
        'recommended_items': [f'item_{i}' for i in range(5)],
    }

    return jsonify(recommended_content)

@app.route('/like-content', methods=['POST'])
def like_content():
    # Get the image and user preferences from the request
    file = request.files['image']
    user_preferences = request.json.get('preferences', {})

    # Preprocess the image
    image = Image.open(io.BytesIO(file.read()))
    image = transform(image).unsqueeze(0)

    # Forward pass through the model
    model.eval()
    with torch.no_grad():
        output = model(image)

    # Simulate recommendation logic
    # (You would replace this with your actual recommendation logic)
    recommended_content = {
        'recommended_images': [f'image_{i}.jpg' for i in range(5)],  # Example recommendations
        'based_on': user_preferences
    }

    return jsonify(recommended_content)

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