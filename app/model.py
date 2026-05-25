import torch
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image
import os

def load_model1():
    # Load the PyTorch ResNet50 model we just trained
    model_path = os.path.join("Resnet50", "Resnet50_PyTorch.pt")
    
    # Recreate the exact architecture from your training script
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 120) # 120 dog breeds
    
    # Load the trained weights (forcing them to CPU for web server inference)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # Set to evaluation mode
    return model

def load_model2():
    # Load the Ultralytics YOLOv8 model
    model_path = os.path.join("yolov8", "20_epoch_model.pt")
    model = YOLO(model_path)
    return model

def preprocess_image_resnet(image_path):
    # PyTorch standard preprocessing (matches what you used in training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0) # Add batch dimension
    return image