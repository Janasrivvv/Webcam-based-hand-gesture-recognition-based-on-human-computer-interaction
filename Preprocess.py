import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def preprocess_image(image):
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Convert the image to RGB
    rgb_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
    
    # Split the RGB channels
    r, g, b = cv2.split(rgb_image)
    
    # Apply adaptive thresholding to each channel
    _, thresholded_r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresholded_g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresholded_b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Merge the thresholded channels back into an RGB image
    thresholded_image = cv2.merge([thresholded_r, thresholded_g, thresholded_b])
    
    # Resize the image to match model input size
    resized_image = cv2.resize(thresholded_image, (64, 64))
    
    # Normalize pixel values to range [0, 1]
    normalized_image = resized_image / 255.0
    
    # Add batch dimension
    reshaped_image = np.expand_dims(normalized_image, axis=0)
    
    return reshaped_image

# def preprocess_image(image):
#     # Apply Gaussian blur to reduce noise
#     blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
#     # Convert the image to RGB
#     rgb_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
    
#     # Resize the image to match model input size
#     resized_image = cv2.resize(rgb_image, (64, 64))
    
#     # Normalize pixel values to range [0, 1]
#     normalized_image = resized_image / 255.0
    
#     return normalized_image 

def preprocess_images_in_directory(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for label in os.listdir(input_dir):
        label_path = os.path.join(input_dir, label)
        if os.path.isdir(label_path):
            for filename in tqdm(os.listdir(label_path), desc=f"Processing {label}"):
                if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                    image_path = os.path.join(label_path, filename)
                    image = cv2.imread(image_path)
                    if image is not None:
                        preprocessed_image = preprocess_image(image)
                        output_path = os.path.join(output_dir, label, filename)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        cv2.imwrite(output_path, (preprocessed_image[0] * 255).astype(np.uint8))  # Convert to uint8 before saving
                    else:
                        print(f"Error loading image: {image_path}")

if __name__ == "__main__":
    input_dir = r"C:\Users\janas\Desktop\PROJ_HCI\Dataset"  # Specify the input directory containing images
    output_dir = r"C:\Users\janas\Desktop\PROJ_HCI\PreprocessedDataset"  # Specify the output directory for preprocessed images
    
    preprocess_images_in_directory(input_dir, output_dir)
    
    print("Preprocessing completed.")
