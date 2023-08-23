# Instructions for Task Implementation

This repository provides instructions and code for implementing two tasks: Task-1 and Task-2. Before getting started, make sure you have set up your environment correctly.

## Setting Up the Environment

### Step 1: Create a Virtual Environment
Create a Python virtual environment to manage dependencies for these tasks. You can do this using the following command:

```bash
python -m venv venv
```

### Step 2: Activate the Virtual Environment (Windows)
Activate the virtual environment by running the following command:

```bash
venv\Scripts\activate
```

## Installing Dependencies

To run the tasks, you need to install the required packages from the `requirements.txt` file. Install these dependencies using pip:

```bash
pip install -r requirements.txt
```

## Task-1: CNN Implementation

### Description

In Task-1, you will develop a Convolutional Neural Network (CNN) model using a provided dataset. The dataset consists of two files:

- `data.csv`: Training data containing accelerometer data (column ACCX) and light conditions (column Light).
- `label.csv`: Labels corresponding to the training data.

### Instructions

1. Execute the Task-1 code by running the following command:

   ```bash
   python task1.py
   ```

2. After the model is trained, the predictions will be saved in `output.txt`.

### Files and Descriptions

- `task1.py`: Python script containing the code for the CNN model.
- `data.csv`: Training data.
- `label.csv`: Labels for the training data.
- `requirements.txt`: File containing all required package dependencies.
- `output.txt`: File containing the model's predictions.

## Task-2: Contrastive Learning Implementation

### Description

Task-2 involves implementing contrastive learning using the accelerometer data. This task has two parts:

- Generating original and augmented images from the ACCX column of `data.csv`.
- Training a contrastive learning model using these images.

### Instructions

1. Generate original and augmented images by running the following commands:

   ```bash
   python part1.py
   ```

   This will create images in the `img` and `aug` folders.

2. Train the contrastive learning model with the following command:

   ```bash
   python part2.py
   ```

   This will use a ResNet18-based Siamese network and two fully connected layers for fine-tuning.

3. The trained encoder's weights will be saved in `encoder_weights_finetuned.pth`, and training progress will be logged in `output2.txt`.

### Files and Descriptions

- `data.csv`: Training data.
- `label.csv`: Labels for the training data.
- `part1.py`: Python script for generating original and augmented images from the ACCX data.
- `part2.py`: Python script for training the contrastive learning model.
- `img/`: Folder containing original images extracted from the ACCX data.
- `aug/`: Folder containing augmented images.
- `requirements.txt`: File containing all required package dependencies.
- `encoder_weights_finetuned.pth`: File with saved encoder weights.
- `output2.txt`: File containing training progress (epochs and loss) of the contrastive learning model.

---

With these enhanced instructions, users should have a clearer understanding of how to set up the environment and run both Task-1 and Task-2.