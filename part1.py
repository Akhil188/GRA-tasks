import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import random

# Load data from CSV
data = pd.read_csv('data.csv')
labels = pd.read_csv('label.csv')  # Load your label.csv
data = data['ACC X'].values
# Reshape data into segments of 2500 rows each
segment_length = 2500
num_segments = len(data) // segment_length
data_segments = np.array_split(data, num_segments)

# Function to generate a spectrogram from a segment
def generate_spectrogram(segment, output_file):
    # Assuming a sample rate of 44100 Hz, you can adjust this based on your data
    sample_rate = 44100

    # Compute the spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(segment.ravel())), ref=np.max)

    # Create a plot of the spectrogram (optional)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(output_file)  # Save the spectrogram as an image

# Function to apply data augmentation to an image
def apply_augmentation(image):
    # Randomly rotate the image by 0, 90, 180, or 270 degrees
    rotation_angle = random.choice([0, 90, 180, 270])
    image = image.rotate(rotation_angle)

    # Randomly flip the image horizontally
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Random cropping to a smaller size (adjust as needed)
    width, height = image.size
    crop_size = min(width, height)
    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))

    return image

# Generate spectrograms and apply data augmentation
for i, segment in enumerate(data_segments):
    # Generate spectrogram
    output_file = f'img/spectrogram_{i}.png'
    generate_spectrogram(segment, output_file)

    # Apply data augmentation to the spectrogram
    input_file = output_file
    output_augmented_file = f'aug/augmented_spectrogram_{i}.png'
    image = Image.open(input_file)
    augmented_image = apply_augmentation(image)
    augmented_image.save(output_augmented_file)
