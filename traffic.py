import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    images = []
    labels = []
    
    for folder in os.listdir(data_dir):
        if folder == ".DS_Store": continue
        else:
            folder_path = os.path.join(data_dir, folder)
            for img in os.listdir(folder_path):
                image_path = os.path.join(folder_path, img)
                read_img = cv2.imread(image_path)
                resize_img = cv2.resize(read_img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(resize_img)
                labels.append(int(folder))

    return np.array(images), np.array(labels)


def get_model():  
    # input layer of the model
    input_array = tf.keras.layers.Input(shape = (IMG_WIDTH, IMG_HEIGHT, 3))
    
    # first convolution and pooling
    conv1 = tf.keras.layers.Conv2D(128, (3, 3), (1, 1), "valid")(input_array)
    conv1 = tf.keras.layers.Conv2D(128, (3, 3), (1, 1), "valid")(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))(conv1)
    dropout1 = tf.keras.layers.Dropout(0.2)(pool1)

    # second convolution and pooling
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), (1, 1), "valid")(dropout1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), (1, 1), "valid")(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size = (1, 1))(conv2)
    dropout2 = tf.keras.layers.Dropout(0.2)(pool2)

    # third convolution and pooling with batch normalization
    conv3 = tf.keras.layers.Conv2D(32, (3, 3), (1, 1), "valid")(dropout2)
    conv3 = tf.keras.layers.Conv2D(32, (3, 3), (1, 1), "valid")(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size = (1, 1))(conv3)
    dropout3 = tf.keras.layers.Dropout(0.3)(pool3)
    batch_norm = tf.keras.layers.BatchNormalization()(dropout3)

    # flatten the layers before dense
    flattened = tf.keras.layers.Flatten()(batch_norm)

    # dense layers for classification 
    dense1 = tf.keras.layers.Dense(128, activation = tf.keras.activations.relu)(flattened)
    dense2 = tf.keras.layers.Dense(64, activation = tf.keras.activations.relu)(dense1)
    dense3 = tf.keras.layers.Dense(NUM_CATEGORIES, activation = tf.keras.activations.softmax)(dense2)
    
    # final constructed model with type "tf.keras.models.Model"
    model = tf.keras.models.Model(inputs = input_array, outputs = dense3)

    # compiling the model
    model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-4),
                  loss = tf.keras.losses.sparse_categorical_crossentropy,
                  metrics = ["acc"])
    
    return model
    
if __name__ == "__main__":
    main()