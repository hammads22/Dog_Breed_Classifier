from flask import Flask, request, render_template, jsonify
import os
import torch
from model import load_model1, load_model2, preprocess_image_resnet

app = Flask(__name__)

# Load both models
model1 = load_model1()
model2 = load_model2()

def load_resnet_classes():
    """
    Dynamically loads the exact 120 class names ResNet was trained on by 
    reading the original DatasetStore folder.
    """
    # Go up one level from 'app' folder, then into DatasetStore/Images
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'DatasetStore', 'Images')
    
    if os.path.exists(dataset_dir):
        # Sort folders identically to how train_resnet_pytorch.py sorted them
        folders = sorted(os.listdir(dataset_dir))
        # Extract the breed name after the hyphen and format it nicely
        return [folder.split('-')[1].replace('_', ' ').title() for folder in folders]
    
    print("WARNING: DatasetStore folder not found. Cannot map ResNet indices.")
    return []

# Load the classes into memory once when the server starts
RESNET_CLASSES = load_resnet_classes()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file
    image_path = os.path.join("static", file.filename)
    file.save(image_path)

    # --- Model 1 (PyTorch ResNet) Prediction ---
    img_resnet = preprocess_image_resnet(image_path)
    
    with torch.no_grad():
        output1 = model1(img_resnet)
        pred1_class_index = torch.argmax(output1, dim=1).item()
    
    # Map index to class name if the DatasetStore was found
    if RESNET_CLASSES and pred1_class_index < len(RESNET_CLASSES):
        pred1_result = RESNET_CLASSES[pred1_class_index]
    else:
        pred1_result = f"Class Index {pred1_class_index}"

    # --- Model 2 (YOLOv8) Prediction ---
    results = model2.predict(image_path)
    pred2_result = "No dog detected"
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        best_box = results[0].boxes[0]
        cls_id = int(best_box.cls[0].item())
        raw_label = results[0].names[cls_id] 
        # Clean up YOLO's raw string to match ResNet's clean formatting
        pred2_result = raw_label.replace('_', ' ').title()

    return jsonify({
        'model1_prediction': pred1_result, 
        'model2_prediction': pred2_result, 
        'image_path': image_path
    })

if __name__ == '__main__':
    app.run(debug=True)