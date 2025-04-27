import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import tensorflow.lite as tflite
from collections import deque
import time

project_root = "/Users/krishnasomani/Documents/Projects/WasteWise"
model_path = os.path.join(project_root, "model", "garbage_classification_model.tflite")

interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]

CONFIDENCE_THRESHOLD = 0.7
SMOOTHING_WINDOW = 5
recent_predictions = deque(maxlen=SMOOTHING_WINDOW)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

labels = ['Biodegradable', 'Non-Recyclable', 'Recyclable', 'Reusable']

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    if mean_brightness < 30:
        label = "No Waste Detected"
        color = (0, 0, 255)
    else:
        resized = cv2.resize(rgb, (input_shape[1], input_shape[0]))
        input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        recent_predictions.append(confidence >= CONFIDENCE_THRESHOLD)

        if sum(recent_predictions) > len(recent_predictions) // 2:
            label = labels[predicted_class]
            color = (0, 255, 0)
        else:
            label = "No Waste Detected"
            color = (0, 0, 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    cv2.putText(frame, label, (20, 40), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Waste Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
