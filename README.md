# Image Depixelation Project
This is a machine learning project for depixelating a (known) area of a grayscale image using a Convolutional Neural Network (CNN).

The goal of the project is to build and train a CNN model. The trained CNN model is then used to predicted pixel values of the pixelated area in a new set of grayscale images with size 64 × 64. Beside the pixelated image, a boolean mask representing the area in the image which is pixelated is passed to the model as input.


This project is intended for learning purposes.


Author: Angelika Vižintin


### Structure
```
example_project
|- architectures.py
|    Class for network architecture.
|- datasets.py
|    Dataset classes and dataset helper functions
|- main.ipynb
|    Juypter notebook including creation of datasets and dataloaders, training & evaluation routines and making predicitions on new data
|- README.md
|    A readme file containing information about the project
|- utils.py
|    Utility function for plotting training and evaluation loss.
|- working_config.json
|     An example configuration file. Can also be done via command line arguments to main.py.
```