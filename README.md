# Dogs vs Cats Classification Using CNN  

This project demonstrates a binary image classification task to distinguish between images of cats and dogs. A Convolutional Neural Network (CNN) is designed and trained using TensorFlow and Keras to achieve this classification.

---

## Project Overview  

- **Objective**: Classify images as either a dog or a cat.  
- **Dataset**: [Dogs vs Cats Dataset from Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats).  
- **Frameworks Used**: TensorFlow, Keras, OpenCV.  
- **Key Techniques**: Image normalization, CNN architecture, training and validation monitoring, and model testing.

---

## Steps to Run the Project  

### 1. **Dataset Setup**  
1. Download the dataset from Kaggle using the API.  
2. Retrieve the Kaggle API token (`kaggle.json`) from your Kaggle account:  
   - Go to your Kaggle account settings and generate the API token.  
   - Upload the `kaggle.json` file to your environment (e.g., Google Colab).  
3. Use the Kaggle CLI to download the dataset.  

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle datasets download -d salader/dogs-vs-cats
```

4. Extract the downloaded ZIP file and organize the data for training and testing.

---

### 2. **Data Preparation**  
1. Load the dataset using `keras.utils.image_dataset_from_directory`.  
   - **Training Dataset**: 20,000 images.  
   - **Validation Dataset**: 5,000 images.  
2. Normalize images to scale pixel values to the range `[0, 1]`.  
3. Define a preprocessing function and apply it to the datasets:  

```python
def process(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)
```

---

### 3. **Model Architecture**  
- The CNN model is built using Keras' Sequential API.  
- Key components:  
  1. **Convolutional Layers**: Extract image features using filters.  
  2. **MaxPooling Layers**: Reduce dimensionality to improve efficiency.  
  3. **Flatten Layer**: Convert 2D data to 1D for Dense layers.  
  4. **Dense Layers**: Fully connected layers for classification.  

#### Model Summary  
```plaintext
1. Conv2D: Extracts features using 32 filters of size 3x3.
2. MaxPooling2D: Reduces feature dimensions.
3. Conv2D: Uses 62 filters of size 3x3.
4. MaxPooling2D: Further reduces feature dimensions.
5. Conv2D: Uses 128 filters of size 3x3.
6. MaxPooling2D: Continues feature reduction.
7. Flatten: Converts data to 1D for Dense layers.
8. Dense: Fully connected layer with 128 neurons.
9. Dense: Fully connected layer with 62 neurons.
10. Dense: Final layer with 1 neuron (sigmoid activation).
```

---

### 4. **Model Compilation and Training**  
- **Loss Function**: `binary_crossentropy` for binary classification.  
- **Optimizer**: Adam.  
- **Metrics**: Accuracy.  

Train the model on the training dataset for 10 epochs and validate using the validation dataset.  

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
```

---

### 5. **Performance Evaluation**  
- Plot training and validation accuracy:  

```python
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.show()
```

- Plot training and validation loss:  

```python
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()
```

---

### 6. **Model Testing**  
1. Load a test image (e.g., `cat.jpg`).  
2. Resize the image to match the input dimensions (256x256).  
3. Make predictions using the trained model.  

```python
test_img = cv2.imread('/content/cat.jpg')
test_img = cv2.resize(test_img, (256, 256))
test_input = test_img.reshape((1, 256, 256, 3))
prediction = model.predict(test_input)
```

- If the output is close to `0`, it predicts a **cat**.  
- If the output is close to `1`, it predicts a **dog**.

---

## Results  
- Training accuracy improves significantly over epochs, but validation accuracy declines due to overfitting.  
- Final accuracy and loss trends indicate potential improvements with data augmentation or regularization.

---

## Libraries Used  
1. **TensorFlow/Keras**: Building and training the CNN.  
2. **OpenCV**: Reading and resizing test images.  
3. **Matplotlib**: Visualizing training and validation performance.

---

## Improvements to Consider  
- Add data augmentation to enhance model generalization.  
- Use dropout layers or regularization to reduce overfitting.  
- Experiment with pre-trained models (e.g., VGG16, ResNet).  

--- 

## Dataset Link  
[Dogs vs Cats Dataset - Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats)  

