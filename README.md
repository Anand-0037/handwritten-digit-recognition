# Handwritten Digit Recognition

A simple neural network project that recognizes handwritten digits (0-9) using TensorFlow and OpenCV.

## Features

- Train on MNIST dataset (60,000 training images)
- Save/load trained models
- Predict custom digit images

## Requirements

Create `requirements.txt`:
```txt
numpy
opencv-python
matplotlib
tensorflow
```

## Installation

## Using uv package manager

### [uv install](https://docs.astral.sh/uv/#highlights)
+ uv init   # to init uv project
+ uv venv   # to create virtual env
+ uv sync   # to install dependencies from pyproject.toml
+ uv run main.py # to run python file

### Windows
```bash
python -m venv digit_env
digit_env\Scripts\activate
pip install -r requirements.txt
```

### Linux/macOS
```bash
python3 -m venv digit_env
source digit_env/bin/activate
pip install -r requirements.txt
```

## Project Structure
```
handwritten-digit-recognition/
├── main.py
├── requirements.txt
├── README.md
├── digits/
│   ├── digit1.png
│   └── digit2.png
└── handwritten-digit.keras
```

## Usage

1. **Train the model:**
   ```bash
   python main.py
   ```

2. **Test with custom images:**
   - Create a `digits/` folder
   - Add PNG images named `digit1.png`, `digit2.png`, etc.
   - You can make using paints in windows for testing purpose.
   - Run the script again

3. **Load saved model:**
   - Uncomment the line: `model = tf.keras.models.load_model("handwritten-digit.model")`
   - Comment out the training code

## Model Architecture

- Input: 28×28 flattened images (784 features)
- Hidden Layer 1: 128 neurons (ReLU)
- Hidden Layer 2: 128 neurons (ReLU)  
- Output: 10 neurons (Softmax)

## Performance

- Training Accuracy: ~98%
- Test Accuracy: ~97%
- Training Time: ~2 minutes

<hr>