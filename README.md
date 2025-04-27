<h1 align="center">WasteWise</h1>
<p align="center" style="margin-top:30px;">
  <img src="https://github.com/user-attachments/assets/b55036c1-89d9-4cb3-bccc-a5d506ba7d56" height="150cm"/>
</p>
The project categorizes waste into biodegradable, non-recyclable, recyclable, and reusable categories using machine learning. It includes data preparation, augmentation, and a convolutional neural network (CNN) model trained on labeled datasets to improve sustainability and waste management.

## Execution Guide:
1. Run the following command line in your terminal:
   ```
   pip install numpy pandas seaborn matplotlib tqdm opencv-python pillow scikit-learn keras tensorflow warnings kaggle
   ```
2. Download the dataset (link to the dataset: https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

3. Copy the path of the dataset folder and paste it into the code

4. After running all the cells, it will create an additional file called `garbage_classification_model.keras` and `garbage_classification_model.tflite` (these files store the model)

5. Enter the path of the image you want in the last cell to check what type of waste it is

6. If you want to integrate the model with the webcame, use the `webcam.py` file and paste the path of the WasteWise folder with the `garbage_classification_model.tflite` model present in the model folder

## Accuracy & Loss Over Epochs:

![image](https://github.com/user-attachments/assets/ec5fa970-7b42-4a72-988a-995bc7d9ea87)

![image](https://github.com/user-attachments/assets/e89d69cc-d585-4a1c-b169-4df1874a25c2)

## Model Predicition:

![image](https://github.com/user-attachments/assets/8b907bea-453c-49d9-99be-37d813cbb93f)

![image](https://github.com/user-attachments/assets/fe14f17d-d482-4905-b262-ffdc3ae2a96e)

![image](https://github.com/user-attachments/assets/2dee0f98-6ae9-47ab-af39-f42f2c7a72e1)

![image](https://github.com/user-attachments/assets/28a76ab1-bd49-4c88-ae98-0cf4ded9d80b)

## Overivew:
Below is the overview of the code:

#### 1. **Library Imports**

Your code imports a wide range of libraries for data processing, model training, and evaluation. Here's a brief explanation of the key libraries:

- **Data Handling**: `os`, `glob`, `shutil`, `time`, `random`, `Path`, `numpy`, `pandas` — Used for file manipulation, numerical computations, data handling, and time-related operations.
- **Visualization**: `seaborn`, `matplotlib.pyplot` — Used for data visualization (plots, charts, etc.).
- **Image Processing**: `cv2`, `PIL.Image` — OpenCV and PIL are used for image processing tasks.
- **Scikit-learn**: `sklearn` — Provides functions for data preprocessing, metrics for evaluation, and machine learning tools such as splitting data, standard scaling, and evaluating models with confusion matrix, classification report, etc.
- **Deep Learning**: `tensorflow.keras` and `keras` — These are the main libraries for building and training deep learning models. Various layers, optimizers, and other Keras utilities are imported to build and train the model.
- **Webcam and Image Handling**: OpenCV (`cv2`) and Keras image utilities (`load_img`, `ImageDataGenerator`) are used for real-time image processing, webcam integration, and image augmentation.

#### 2. **Image Preprocessing and Augmentation**

**ImageDataGenerator**: This is used for augmenting images to prevent overfitting. Common transformations (like rotations, shifts, and horizontal flips) are applied to the images during model training to artificially increase the dataset size.

#### 3. **Model Building and Training**

Convolutional Neural Network (CNN) is being used to built with Keras layers like `Conv2D`, `MaxPool2D`, `Dense`, `Dropout`, and `BatchNormalization`. These layers are used to process images for classification tasks. Below is a likely model-building strategy:

- **Convolutional Layers**: Apply convolution operations to extract features from images.
- **Pooling Layers**: Used to reduce the spatial dimensions (downsample) of the feature maps.
- **Dense Layers**: Fully connected layers that help in classification based on the learned features.
- **Dropout Layers**: Prevent overfitting by randomly setting some neurons to zero during training.
- **BatchNormalization**: Normalizes the output of a previous activation layer, improving model performance.

#### 4. **Compiling the Model**

It compiles the model with an optimizer (like `Adam`, `SGD`, or `RMSprop`), loss function (e.g., `categorical_crossentropy`), and metrics (e.g., accuracy). The model is then ready to be trained.

#### 5. **Webcam Integration for Real-Time Detection**

- **Webcam Feed**: The code captures a continuous feed from the webcam using OpenCV (`cv2.VideoCapture`).
- **Image Preprocessing**: For each frame from the webcam, the image is resized, preprocessed, and passed through the model for classification. You likely use `load_img()` to load and preprocess the frame.
- **Model Prediction**: The frame is passed through the trained CNN model, and the predicted class (fruit) is determined.
- **Bounding Boxes**: If a fruit is detected, you use OpenCV's `cv2.rectangle()` to draw bounding boxes around the detected fruit.
- **Real-time Display**: The result is shown on the screen in real-time using `cv2.imshow()`, allowing users to see the webcam feed with the detected fruits highlighted.

#### 6. **Evaluation and Metrics**

Once the model is trained, you evaluate its performance using various metrics from Scikit-learn, such as:

- **Confusion Matrix**: Displays how well the model classified each class.
- **Accuracy Score**: Percentage of correctly classified instances.
- **Precision, Recall, F1-Score**: Key metrics to evaluate the model, especially for imbalanced classes.
- **ROC Curve**: Plots the true positive rate versus the false positive rate to evaluate classification performance.

#### 7. **Callbacks for Model Training**

- **EarlyStopping**: Monitors a performance metric (e.g., validation loss) during training and stops training if it stops improving, preventing overfitting.
- **ModelCheckpoint**: Saves the model at certain intervals (e.g., after each epoch) to keep the best model based on validation performance.
- **ReduceLROnPlateau**: Reduces the learning rate if the validation performance plateaus.

#### 8. **Class Weighting (Optional)**

You may use **`compute_class_weight`** from Scikit-learn if your dataset has class imbalance. This assigns higher weight to underrepresented classes to balance the contribution of each class during training.
