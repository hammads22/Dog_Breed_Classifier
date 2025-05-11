Dog Breed Classification and Detection System
=============================================

This project is a web-based application for identifying dog breeds using two deep learning models:
- YOLOv8: For object detection (detects dogs in the image)
- ResNet50: For image classification (predicts the breed of the dog)

Google Colab Notebook:
https://colab.research.google.com/drive/1xyCztxC3i8KgjJbprxEhW82JOfGjrcm5?usp=sharing

Models Used
-----------
1. YOLOv8
   - Task: Detect dogs in an image
   - Output: Bounding boxes

2. ResNet50
   - Task: Classify breed of dog from image
   - Output: Breed label (e.g., Golden Retriever, German Shepherd, etc.)

How It Works
------------
1. User uploads an image through the UI.
2. Selects which model to use:
   - YOLOv8 → Detects dog(s) in the image with bounding boxes
   - ResNet50 → Classifies the breed of the dog
3. Results are displayed on the page.

Run Locally
-----------
1. Clone the repo and navigate into it
2. Set up a virtual environment and activate it
3. Install dependencies
4. Run the Flask app

Example:
------------------
git clone https://github.com/yourusername/dog-breed-detection-app.git
cd dog-breed-detection-app
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py

Sample Output
-------------
YOLOv8 → Image with dog detection bounding boxes
ResNet50 → Dog breed: Labrador Retriever

Future Work
-----------
- Improve breed classification with better cropping and fine-tuning
- Add batch processing and multiple dog detection
- Explore mobile-friendly deployment

Acknowledgements
----------------
- YOLOv8 by Ultralytics
- ResNet50 by Kaiming He et al.
- Dataset from Stanford Dogs (http://vision.stanford.edu/aditya86/ImageNetDogs/)
