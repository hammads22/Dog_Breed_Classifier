```markdown
# 🐕 AI Dog Breed Analyzer

An elegant, web-based artificial intelligence vision application that utilizes dual deep-learning models to analyze images. The system detects the presence of dogs using YOLOv8 and precisely classifies them into one of 120 distinct breeds using a custom-trained PyTorch ResNet50 model.


## ✨ Features

* **Dual AI Pipeline:** Runs inference simultaneously on two distinct neural networks.
* **High-Precision Classification:** Identifies 120 specific dog breeds using the Stanford Dogs dataset.
* **Hardware-Optimized Backend:** Custom PyTorch implementation featuring Automatic Mixed Precision (AMP), pinned memory, and CuDNN benchmarking for rapid inference.
* **Modern User Interface:** A sleek, dark-themed glassmorphism frontend with real-time image previews and asynchronous loading states.
* **Containerized Deployment:** Fully configured Docker support for guaranteed cross-platform execution.



## 🧠 Models Used

### 1. ResNet50 (PyTorch)
* **Task:** Fine-grained image classification.
* **Architecture:** Pre-trained ResNet50 backbone with a custom fully connected head fine-tuned on the Stanford Dogs dataset.
* **Output:** Precise breed string label (e.g., "Welsh Springer Spaniel").

### 2. YOLOv8 (Ultralytics)
* **Task:** Object detection and localization.
* **Output:** Identifies the presence of a dog in the frame and provides a confidence score.

*(Note: The Google Colab notebook for the original data exploration and initial training phase can be found [here](https://colab.research.google.com/drive/1xyCztxC3i8KgjJbprxEhW82JOfGjrcm5?usp=sharing)).*


## 🚀 Installation & Setup

### Prerequisites
* Python 3.10+
* Git

### 1. Clone the Repository
```bash
git clone [https://github.com/hammads22/Dog_Breed_Classifier.git](https://github.com/hammads22/Dog_Breed_Classifier.git)
cd Dog_Breed_Classifier

```

### 2. Set Up the Virtual Environment

```powershell
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

```

### 3. Install Dependencies

This project provides two separate requirements files depending on your hardware:

**For NVIDIA GPU Acceleration (Recommended for local development):**

```bash
pip install -r requirements-gpu.txt

```

**For CPU Only (Recommended for standard web server deployment):**

```bash
pip install -r requirements-cpu.txt

```

---

## 💻 Running the Application

### Option A: Standard Execution

Once your environment is active and dependencies are installed, navigate to the `app` directory and start the Flask server:

```bash
cd app
python app.py

```

Open your web browser and navigate to `http://127.0.0.1:5000`.

### Option B: Docker Deployment

If you prefer running the application in a sandboxed container, ensure Docker is running and execute the following:

```bash
# Build the image
docker build -t dog-breed-classifier .

# Run the container
docker run -p 5000:5000 dog-breed-classifier

```

The app will be live at `http://localhost:5000`.

---

## 🛠️ Project Structure

The project maintains a strict separation of concerns, keeping training pipelines distinct from the production web application:

* `app/`: Contains the Flask server, routing logic, and the pre-trained `.pt` model weights.
* `static/`: Temporary storage for user-uploaded images awaiting AI inference.
* `templates/`: Contains the modern HTML/CSS/JS frontend interface.
* `training_scripts/`: The optimized PyTorch and YOLOv8 training pipelines used to build the models.

---

## 📈 Future Work

* Improve classification accuracy by utilizing the YOLOv8 bounding box to crop the image before feeding it into the ResNet50 model.
* Add batch processing capabilities for analyzing multiple dogs in a single image.
* Explore converting the PyTorch model to ONNX format for even faster CPU inference times on edge devices.

---

## 🙏 Acknowledgements

* **YOLOv8** by [Ultralytics](https://www.google.com/search?q=https://ultralytics.com/)
* **ResNet50 Architecture** by Kaiming He et al.
* **Dataset** provided by [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)

```

```