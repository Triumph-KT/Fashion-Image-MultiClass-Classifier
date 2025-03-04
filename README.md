# Fashion MNIST Apparel Classification Using Artificial Neural Networks (ANN)

Developed an artificial neural network (ANN) to classify grayscale images of fashion items from the Fashion MNIST dataset, demonstrating the impact of model architecture, learning rate adjustments, and hyperparameter tuning on multi-class image classification performance.

## Project Overview

This project applies machine learning techniques to the Fashion MNIST dataset, which consists of clothing item images. By progressively refining an ANN architecture, the model accurately classifies apparel into 10 distinct categories, supporting real-world applications in computer vision and retail analytics.

## Dataset

The **Fashion MNIST dataset** contains:
- 60,000 training images and 10,000 test images.
- Each image is a **28x28 grayscale pixel** representation.
- 10 apparel classes:
  - 0 – T-shirt/top
  - 1 – Trouser
  - 2 – Pullover
  - 3 – Dress
  - 4 – Coat
  - 5 – Sandal
  - 6 – Shirt
  - 7 – Sneaker
  - 8 – Bag
  - 9 – Ankle boot

## Objectives

- Preprocess image data and apply normalization.
- Build and compare multiple ANN architectures.
- Optimize model performance by adjusting layers and hyperparameters.
- Evaluate models using precision, recall, and accuracy.
- Visualize misclassifications and interpret performance insights.

## Methods

### Data Preprocessing:
- Normalized pixel values to range [0, 1].
- One-hot encoded the categorical target labels.
- Visualized sample images for exploratory analysis.

### Model Development:
Trained and evaluated three ANN models:
- **Model 1**:
  - Single hidden layer with 64 neurons and ReLU activation.
  - 10-neuron softmax output for multi-class classification.
  - Trained for **10 epochs**.
- **Model 2**:
  - Same architecture as Model 1.
  - Increased to **30 epochs** and modified the learning rate to **0.001**.
- **Model 3**:
  - Added a hidden layer with **128 neurons**.
  - Followed by a 64-neuron layer and softmax output.
  - Trained for **30 epochs**.

### Evaluation:
- Accuracy, precision, recall, and F1-score.
- Confusion matrices to interpret misclassification patterns.
- Visualization of predicted vs. actual labels on test images.

## Results

- **Model 1**: ~87% accuracy on test data.
- **Model 2**: Improved performance with longer training but slight overfitting risk.
- **Model 3**: Best performance with ~89-90% test accuracy.
  - Improved recognition of visually complex classes.
  - Class **6 (Shirt)** remained the most challenging due to visual similarity with other tops.
- Identified **class overlap** challenges, notably between T-shirts, shirts, pullovers, and coats.
- Model improvements effectively reduced misclassification and improved precision.

## Business/Scientific Impact

- Demonstrated the applicability of ANNs in classifying fashion apparel from raw pixel data.
- Supports automated tagging and sorting of clothing items in e-commerce or inventory management systems.
- Provides a scalable approach for initial classification tasks prior to deploying more advanced architectures like CNNs.

## Technologies Used

- Python
- TensorFlow (Keras)
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/fashion-mnist-ann-classification.git
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4. Open and run the notebook to:
   - Preprocess the dataset.
   - Train and compare multiple ANN models.
   - Evaluate the performance of each model.
   - Visualize predictions and misclassifications.

## Future Work

- Integrate **Convolutional Neural Networks (CNNs)** for improved spatial feature extraction.
- Apply **dropout regularization** to reduce overfitting.
- Experiment with advanced optimizers and learning rate schedules.
- Conduct **hyperparameter tuning** for layer sizes, activation functions, and batch sizes.
- Extend the model to handle additional datasets or multi-label classification scenarios.
