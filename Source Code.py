from keras.preprocessing.image import ImageDataGenerator

# Image dimensions
img_width, img_height = 128, 128

train_data_dir = 'C:\\Users\\HP\\Desktop\\FIRE DETECTION SYSTEM\\train_data'
validation_data_dir = 'C:\\Users\\HP\\Desktop\\FIRE DETECTION SYSTEM\\validation_data'

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary')
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the model
model = create_model()

# Save the model after every epoch
checkpointer = ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True)

# Train the model
model.fit(
    train_generator,
    epochs=3,
    validation_data=validation_generator,
    callbacks=[checkpointer])
from keras.models import load_model
import cv2
import numpy as np

# Load the model from disk
model = load_model('C:\\Users\\HP\\Desktop\\FIRE DETECTION SYSTEM\\model.h5')

def predict(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Use the model to predict if there is fire in the image
    prediction = model.predict(img)
    return prediction[0][0]
