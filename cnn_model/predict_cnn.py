import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = SimpleCNN()
model.load_state_dict(torch.load('cnn_model/model_cnn_pytorch.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    return output.item()

def predict_with_label(img_path, threshold=0.5):
    score = predict_image(img_path)
    label = "AI-Generated" if score > threshold else "Real"
    return label, round(score, 3)
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Please provide an image path as argument.")
    else:
        img_path = sys.argv[1]
        label, score = predict_with_label(img_path)
        print(f"Prediction: {label} (Confidence: {score})")
