import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Directories
defective_dir = 'data/defective'
non_defective_dir = 'data/non_defective'

# Image size for resizing
img_size = (128, 128)

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    return images, labels

defective_images, defective_labels = load_images_from_folder(defective_dir, 1)
non_defective_images, non_defective_labels = load_images_from_folder(non_defective_dir, 0)

# Combine the data
images = np.array(defective_images + non_defective_images)
labels = np.array(defective_labels + non_defective_labels)

# Normalize the images
images = images / 255.0
images = images.reshape((images.shape[0], -1))  # Flatten images

# Split the data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dimensionality reduction with PCA
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
