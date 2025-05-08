<h1 align="center">WasteWise</h1>
<p align="center" style="margin-top:30px;">
  <img src="https://github.com/user-attachments/assets/b55036c1-89d9-4cb3-bccc-a5d506ba7d56" height="150cm"/>
</p>
The project categorizes waste into biodegradable, non-recyclable, recyclable, and reusable categories using machine learning. It includes data preparation, augmentation, and a convolutional neural network (CNN) model trained on labeled datasets to improve sustainability and waste management.

## Execution Guide:
1. Run the following command line in your terminal:
   ```
   git clone https://github.com/kr1shnasomani/WasteWise.git
   cd WasteWise
   ```

2. Download the dependencies:
   ```
   pip install -r requirements
   ```

3. Run the code and it will create an additional file called `garbage_classification_model.keras` and `garbage_classification_model.tflite` (these files store the model)

4. If you want to integrate the model with the webcame, use the `webcam.py` file and paste the path of the WasteWise folder with the `garbage_classification_model.tflite` model present in the model folder

## Accuracy & Loss Over Epochs:

![image](https://github.com/user-attachments/assets/ec5fa970-7b42-4a72-988a-995bc7d9ea87)

![image](https://github.com/user-attachments/assets/e89d69cc-d585-4a1c-b169-4df1874a25c2)

## Model Predicition:

![image](https://github.com/user-attachments/assets/8b907bea-453c-49d9-99be-37d813cbb93f)

![image](https://github.com/user-attachments/assets/fe14f17d-d482-4905-b262-ffdc3ae2a96e)

![image](https://github.com/user-attachments/assets/2dee0f98-6ae9-47ab-af39-f42f2c7a72e1)

![image](https://github.com/user-attachments/assets/28a76ab1-bd49-4c88-ae98-0cf4ded9d80b)
