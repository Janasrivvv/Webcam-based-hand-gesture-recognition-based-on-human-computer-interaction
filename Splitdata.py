import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(input_dir, output_dir, test_size=0.1, val_size=0.1, random_state=42):
    # Create output directories if they don't exist
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    val_dir = os.path.join(output_dir, 'val')
    for directory in [train_dir, test_dir, val_dir]:
        os.makedirs(directory, exist_ok=True)
    
    for label in os.listdir(input_dir):
        label_path = os.path.join(input_dir, label)
        if os.path.isdir(label_path):
            # Split images for the current label
            images = os.listdir(label_path)
            X_train, X_test_val = train_test_split(images, test_size=test_size+val_size, random_state=random_state)
            X_val, X_test = train_test_split(X_test_val, test_size=val_size/(test_size+val_size), random_state=random_state)
            
            # Copy images to train, test, and val directories
            for filename in X_train:
                src = os.path.join(label_path, filename)
                dst = os.path.join(train_dir, label)
                os.makedirs(dst, exist_ok=True)
                shutil.copy(src, dst)
            for filename in X_test:
                src = os.path.join(label_path, filename)
                dst = os.path.join(test_dir, label)
                os.makedirs(dst, exist_ok=True)
                shutil.copy(src, dst)
            for filename in X_val:
                src = os.path.join(label_path, filename)
                dst = os.path.join(val_dir, label)
                os.makedirs(dst, exist_ok=True)
                shutil.copy(src, dst)

if __name__ == "__main__":
    input_dir = r"C:\Users\janas\Desktop\PROJ_HCI\PreprocessedDataset1"  # Specify the input directory containing preprocessed images
    output_dir = r"C:\Users\janas\Desktop\PROJ_HCI\SplitDataset"  # Specify the output directory for split dataset
    
    split_dataset(input_dir, output_dir)
    
    print("Dataset split completed.")
import os

def count_images_in_directory(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                count += 1
    return count

if __name__ == "__main__":
    output_dir = r"C:\Users\janas\Desktop\PROJ_HCI\SplitDataset"  # Specify the output directory for split dataset
    
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    val_dir = os.path.join(output_dir, 'val')
    
    num_train_images = count_images_in_directory(train_dir)
    num_test_images = count_images_in_directory(test_dir)
    num_val_images = count_images_in_directory(val_dir)
    
    print("Number of images in train directory:", num_train_images)
    print("Number of images in test directory:", num_test_images)
    print("Number of images in val directory:", num_val_images)
