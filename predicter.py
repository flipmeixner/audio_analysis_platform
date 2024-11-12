import redis
import pickle
import time
import torch
import torch.nn as nn

# CNN Model Definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(19968, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # Calculate flattened size dynamically
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.view(x.size(0), -1).shape[1], 64).to(x.device)

        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def prediction_script():
    r = redis.Redis(host='localhost', port=6379, db=0)
    model = SimpleCNN()
    model.load_state_dict(torch.load('cnn_model.pth', weights_only=True))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to appropriate device

    previous_keys = set()

    while True:
        keys = set(r.keys('mel_spec:*'))
        new_keys = keys - previous_keys
        for key in new_keys:
            mel_spec_db = pickle.loads(r.get(key))
            mel_spec_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            # Make prediction
            with torch.no_grad():  # Disable gradient calculation for inference
                prediction = model(mel_spec_tensor)
                prediction_score = prediction[0][0].item()  # Get the scalar prediction score

            # Store prediction score in Redis
            timestamp = key.decode("utf-8").split(":")[1]
            r.set(f'prediction:{timestamp}', prediction_score)

        # Update previous keys and sleep briefly
        previous_keys = keys
        time.sleep(0.01)

if __name__ == '__main__':
    prediction_script()
