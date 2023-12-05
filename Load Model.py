from keras.models import load_model
import cv2
import numpy as np

# Load the model from disk
model = load_model('C:\\Users\\HP\\Desktop\\FIRE DETECTION SYSTEM\\model.h5')

def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    # Resize image
    img = cv2.resize(img, (128, 128))
    # Normalize image
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)

    # Use the model to predict if there is fire in the image
    prediction = model.predict(img)
    
    # Assuming your model outputs a single sigmoid unit, and you threshold at 0.5
    if prediction[0] > 0.5:
        return "Fire detected"
    else:
        return "No fire detected"

# Assume you have an image 'test.jpg'
image_path = 'test.jpg.png'

# Make a prediction
result = predict(image_path)

print(result)
