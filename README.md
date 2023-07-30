# MNIST LeNet-like Convolutional Neural Network

This repository contains an implementation of a LeNet-like Convolutional Neural Network (CNN) for the MNIST dataset classification task using Keras and TensorFlow.
![image](https://github.com/FarzadMalik/LeNet_mnist_KERAS/assets/107833662/7090a80a-b920-484c-9a0d-64bf1c2873aa)

## Dependencies

Before running the code in this repository, ensure you have the following dependencies installed:

- Keras (Version 2.6.0)
- TensorFlow (Version 2.6.0)
- Matplotlib (Version 3.4.3)

You can install the required packages using the following command:

```
pip install -r requirements.txt
```

## Introduction

The MNIST dataset consists of handwritten digits from 0 to 9 and is widely used for image classification tasks. In this repository, we build a LeNet-like CNN to classify the images in the MNIST dataset.

## Model Architecture

| Layer Type                 | Output Shape          | Number of Parameters |
|----------------------------|-----------------------|----------------------|
| Input                      | (28, 28, 1)           | 0                    |
| Conv2D (6 filters)         | (28, 28, 6)           | 156                  |
| ReLU                       | (28, 28, 6)           | 0                    |
| MaxPooling2D               | (14, 14, 6)           | 0                    |
| Conv2D (16 filters)        | (14, 14, 16)          | 2416                 |
| ReLU                       | (14, 14, 16)          | 0                    |
| MaxPooling2D               | (7, 7, 16)            | 0                    |
| Conv2D (120 filters)       | (7, 7, 120)           | 48120                |
| ReLU                       | (7, 7, 120)           | 0                    |
| MaxPooling2D               | (3, 3, 120)           | 0                    |
| Flatten                    | (1080,)               | 0                    |
| Dense (120 neurons)        | (120,)                | 129720               |
| ReLU                       | (120,)                | 0                    |
| Dense (84 neurons)         | (84,)                 | 10164                |
| ReLU                       | (84,)                 | 0                    |
| Dense (10 neurons)         | (10,)                 | 850                  |
| Softmax                    | (10,)                 | 0                    |

Total Number of Parameters: 191,426

Note: The number of parameters is computed based on the formula `(filter_height * filter_width * input_channels + 1) * number_of_filters`, where `+1` is for the bias term. For the Flatten layer, no parameters are involved in the transformation.
The LeNet-like CNN architecture consists of the following layers:

1. Input layer: Convolutional layer with 6 filters of size (5, 5), followed by ReLU activation and MaxPooling with pool size (2, 2) and stride (2, 2).
2. Convolutional layer: 16 filters of size (5, 5), followed by ReLU activation and MaxPooling with pool size (2, 2) and stride (2, 2).
3. Convolutional layer: 120 filters of size (5, 5), followed by ReLU activation and MaxPooling with pool size (2, 2) and stride (2, 2).
4. Flatten layer to convert the 3D output to 1D.
5. Fully connected layer: 120 neurons with ReLU activation.
6. Fully connected layer: 84 neurons with ReLU activation.
7. Output layer: Dense layer with 10 neurons (equal to the number of classes in the MNIST dataset) and a softmax activation function for classification.

## Data Preprocessing

The MNIST dataset is loaded using the Keras `mnist.load_data()` function. The images are reshaped and normalized to values between 0 and 1.

## Training

The model is trained for 50 epochs using the Adadelta optimizer and categorical cross-entropy loss. The training and validation loss and accuracy are plotted over epochs.

## Files

- `mnist_lenet.ipynb`: Jupyter Notebook containing the implementation and training code.
- `mnist_LeNet.h5`: Trained model saved in h5 format.

## Usage

To run the code, simply open the `mnist_lenet.ipynb` Jupyter Notebook and execute the cells. Ensure that you have installed the required dependencies as mentioned in the `Dependencies` section.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to use, modify, and distribute the code for your own projects.

---

Happy coding! If you have any questions or suggestions, feel free to open an issue or contact me.

Author: Farzad Malik
Email: anpmlk6@qq.com
GitHub: [FarzadMalik](https://github.com/farzadMalik)
