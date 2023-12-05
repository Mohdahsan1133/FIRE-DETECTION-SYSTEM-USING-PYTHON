# FIRE-DETECTION-SYSTEM-USING-PYTHON
Using deep learning and image processing techniques to detect fire in images as a supplementary tool to traditional fire detection systems AI-Based Fire Detection System This repository contains code for a deep learning project aimed at detecting fire in images. The system can be used as a supplementary tool to enhance traditional fire detection systems.

The project uses Python, Keras, TensorFlow, and OpenCV to build and train a Convolutional Neural Network (CNN) on a dataset of images categorized into two classes - fire and no-fire.

The repository is organized as follows:

train.py - This script prepares the image data, builds and trains the model, and saves the trained model. predict.py - This script loads the trained model and uses it to predict whether a given image contains fire or not. Setup and Installation Clone this repository.

Install the required packages: pip install tensorflow keras opencv-python-headless numpy Pillow Place your dataset in the appropriate directory and specify the paths in train.py.

Run-train.py to train the model.

Run-predict.py to use the trained model for prediction. Usage To train the model: python train.py To predict with the model: python predict.py --image_path test.jpg Note This AI-based fire detection system is designed to be used as a supplemental tool to traditional fire detection systems, and not as a standalone solution.

Remember to replace all placeholder text with your actual file names, directory names, etc., and add any additional information relevant to your project.
