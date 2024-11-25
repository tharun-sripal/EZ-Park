import cv2
import numpy as np
import os

# Preprocess function to resize the image and normalize
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize the image
    return image

# Example of loading a dataset
def load_dataset(dataset_path):
    images = []
    labels = []
    for filename in os.listdir(dataset_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(dataset_path, filename)
            image = preprocess_image(image_path)
            images.append(image)
            # Assume labels are in the filename like 'ABC1234.jpg'
            label = filename.split('.')[0]
            labels.append(label)
    return np.array(images), np.array(labels)

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten
from keras.layers import Reshape
from keras.optimizers import Adam

def build_model(input_shape=(224, 224, 1), num_classes=36):  # For alphanumeric characters
    # Input layer
    input_layer = Input(shape=input_shape)

    # CNN layers for feature extraction
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)

    # Reshaping for LSTM
    x = Reshape((-1, 128))(x)

    # LSTM layers
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)

    # Output layer
    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

from keras.utils import to_categorical

# Load your dataset
dataset_path = 'path_to_your_number_plate_images'
images, labels = load_dataset(dataset_path)

# Prepare the labels (one-hot encoding for classification)
# Assuming the labels are strings with alphanumeric characters
all_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
char_to_index = {char: idx for idx, char in enumerate(all_chars)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

# Convert labels to one-hot encoding
labels_encoded = [to_categorical([char_to_index[char] for char in label], num_classes=len(all_chars)) for label in labels]

# Train the model
model = build_model()

model.fit(images, np.array(labels_encoded), epochs=10, batch_size=32)

# Save the model
model.save('your_plate_recognition_model.h5')

from keras.models import load_model

# Load the pre-trained model
model = load_model('your_plate_recognition_model.h5')

# Inference function
def recognize_number_plate(frame, model):
    image = preprocess_image(frame)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)

    # Decode predictions (e.g., convert to string)
    plate_number = ''.join([index_to_char[np.argmax(p)] for p in predictions[0]])
    return plate_number

