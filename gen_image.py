import os
import numpy as np
import matplotlib.pyplot as plt
from umbmid.loadsave import load_pickle

# Define the path to the clean dataset directory
DATA_DIR = os.path.join('datasets', 'gen-one', 'clean')

# Load the frequency-domain data
train_data = load_pickle(os.path.join(DATA_DIR, 'train_data.pickle'))

# Select the first sample for reconstruction
sample_freq_data = train_data[0]

# Function to reconstruct and reshape a single image from frequency domain data
def reconstruct_image(freq_data, target_shape=(100, 200)):
    # Apply 2D inverse Fourier transform
    reconstructed = np.fft.ifft2(freq_data, axes=(0, 1))
    
    # Take the magnitude (absolute value) to get the real part of the image
    spatial_image = np.abs(reconstructed)
    
    # Reshape the image to the specified rectangular target shape
    # reshaped_image = np.reshape(spatial_image, target_shape)
    
    return spatial_image

# Set the desired rectangular shape
target_shape = (100, 200)  # Adjust this as needed

# Reconstruct and reshape the sample image
spatial_image = reconstruct_image(sample_freq_data, target_shape)

# Display the reconstructed image
plt.imshow(spatial_image, cmap='gray')
plt.title('Reconstructed Image from Frequency Domain')
plt.axis('off')
plt.show()
plt.savefig('cancer.jpg')
