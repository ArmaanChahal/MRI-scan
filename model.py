import os
import numpy as np
import tensorflow as tf
import boto3
import shutil
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# AWS / S3 Configuration
# ----------------------------
S3_BUCKET = 'cmpt-340-rownak-merged'
S3_TRAIN_PREFIX = 'merged_output/Training/'
S3_TEST_PREFIX  = 'merged_output/Testing/'

# Directories on S3 contain four subfolders:
#   merged-glioma, merged-meningioma, merged-pituitary, merged-no_tumor

# Local directories where we will download the images
LOCAL_TRAIN_DIR = '/tmp/training'
LOCAL_TEST_DIR = '/tmp/testing'

# Directories within S3 where we will upload outputs
S3_MODEL_DIR = 'model_output'
S3_RESULTS_DIR = 'results_output'

# Create S3 client (ensure IAM role or credentials are configured)
s3_client = boto3.client('s3')

# ----------------------------
# Download Function
# ----------------------------
def download_s3_folder(bucket, s3_prefix, local_dir):
    """
    Downloads all objects under a given S3 prefix to a local directory.
    """
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                # Skip if the key is a "folder" (ends with '/')
                if key.endswith('/'):
                    continue
                # Create a corresponding local directory structure
                rel_path = os.path.relpath(key, s3_prefix)
                local_path = os.path.join(local_dir, rel_path)
                local_folder = os.path.dirname(local_path)
                os.makedirs(local_folder, exist_ok=True)
                if not os.path.exists(local_path):
                    print(f"Downloading s3://{bucket}/{key} to {local_path}")
                    s3_client.download_file(bucket, key, local_path)
                    
# Download training and testing data locally
download_s3_folder(S3_BUCKET, S3_TRAIN_PREFIX, LOCAL_TRAIN_DIR)
download_s3_folder(S3_BUCKET, S3_TEST_PREFIX, LOCAL_TEST_DIR)

# ----------------------------
# Training Parameters
# ----------------------------
IMAGE_SIZE = 128
BATCH_SIZE = 20
STAGE1_EPOCHS = 5    # Stage 1: Train classifier layers (frozen base)
STAGE2_EPOCHS = 10   # Stage 2: Fine-tune last layers

# ----------------------------
# Helper Functions for Data Preprocessing
# ----------------------------
def get_class_names(local_dir):
    """
    Lists subfolders in the local_dir as class names.
    """
    return sorted([d for d in os.listdir(local_dir) if os.path.isdir(os.path.join(local_dir, d))])

def get_file_paths_and_labels(local_dir, class_names):
    """
    Walk through each class folder to collect file paths and assign labels.
    """
    file_paths = []
    labels = []
    for label in class_names:
        folder = os.path.join(local_dir, label)
        for fname in os.listdir(folder):
            # Optional: filter files if needed (e.g., based on prefix "tr" for training, "te" for testing)
            file_paths.append(os.path.join(folder, fname))
            labels.append(label)
    return shuffle(file_paths, labels)

def preprocess_train(path, label):
    # Read image file from local path
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    # Data augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def preprocess_test(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def encode_label(label_list, class_names):
    return np.array([class_names.index(lbl) for lbl in label_list])

# ----------------------------
# 1) Prepare Local Data
# ----------------------------
print("Preparing local training and testing data...")
train_class_names = get_class_names(LOCAL_TRAIN_DIR)
test_class_names = get_class_names(LOCAL_TEST_DIR)
print("Training class names:", train_class_names)
print("Testing class names:", test_class_names)

# (Assuming the class names match between train and test)
class_names = train_class_names

# Get file paths and labels
train_paths, train_labels = get_file_paths_and_labels(LOCAL_TRAIN_DIR, class_names)
test_paths, test_labels = get_file_paths_and_labels(LOCAL_TEST_DIR, class_names)

# Encode labels to integer indices
encoded_train_labels = encode_label(train_labels, class_names)
encoded_test_labels = encode_label(test_labels, class_names)

# Compute class weights to handle class imbalance
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(encoded_train_labels),
    y=encoded_train_labels
)
class_weights_dict = dict(enumerate(class_weights_array))
print("Class weights:", class_weights_dict)

# Create TensorFlow Datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, encoded_train_labels))
train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_paths, encoded_test_labels))
test_ds = test_ds.map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ----------------------------
# 2) Build the Model (Using ResNet50 as Base)
# ----------------------------
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
base_model.trainable = False  # Freeze the base model initially

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# ----------------------------
# 3) Stage 1 Training: Train Classifier Layers (Frozen Base)
# ----------------------------
print(f"\n--- Stage 1: Training (Frozen Base) for {STAGE1_EPOCHS} epochs ---")
history_stage1 = model.fit(
    train_ds,
    epochs=STAGE1_EPOCHS,
    class_weight=class_weights_dict
)

# ----------------------------
# 4) Stage 2: Fine-Tune the Last 20 Layers of the Base Model
# ----------------------------
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=2)
]

print(f"\n--- Stage 2: Fine-tuning last 20 layers for {STAGE2_EPOCHS} epochs ---")
history_stage2 = model.fit(
    train_ds,
    epochs=STAGE2_EPOCHS,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# ----------------------------
# 5) Evaluate the Model & Generate Reports
# ----------------------------
preds = model.predict(test_ds)
predicted_classes = np.argmax(preds, axis=1)

cm = confusion_matrix(encoded_test_labels, predicted_classes)
print("\nConfusion Matrix:")
print(cm)

report = classification_report(encoded_test_labels, predicted_classes, target_names=class_names)
print("\nClassification Report:")
print(report)

with open('confusion_matrix.txt', 'w') as f:
    f.write("Confusion Matrix:\n")
    f.write(str(cm))

with open('classification_report.txt', 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)

# ----------------------------
# 6) Upload Results and Model Back to S3
# ----------------------------
print("\nUploading results and model back to S3...")

s3_client.upload_file('confusion_matrix.txt', S3_BUCKET, f'{S3_RESULTS_DIR}/confusion_matrix.txt')
s3_client.upload_file('classification_report.txt', S3_BUCKET, f'{S3_RESULTS_DIR}/classification_report.txt')

# Save the model locally
model.save('mri_model.keras')
# Zip the model directory (if model is saved as a folder)
shutil.make_archive('mri_model', 'zip', 'mri_model.keras')
s3_client.upload_file('mri_model.zip', S3_BUCKET, f'{S3_MODEL_DIR}/mri_model.zip')

print("\nModel and results have been successfully uploaded to S3.")
print(f"Model: s3://{S3_BUCKET}/{S3_MODEL_DIR}/mri_model.zip")
print(f"Reports: s3://{S3_BUCKET}/{S3_RESULTS_DIR}/")