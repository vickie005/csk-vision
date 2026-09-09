import cv2
import tensorflow as tf
import numpy as np
import time
from collections import Counter

#loading the existing trained model
model = tf.keras.models.load_model("csk_vision_model.keras")

class_names = ["approved", "peace", "wave"]

prediction_history = []
start_time = time.time()
prediction_duration = 1.0
camera = cv2.VideoCapture(0)

label = "Analyzing..." # coz label won't exist during the first 1.0 seconds
while True:
    success, frame = camera.read()

    if not success:
        print("Could not access the webcam.")
        break

    image = cv2.resize(frame, (224, 224))     #resizing webcam frame to model input size
    # convert to RGB (OpenCV uses BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # then convert image into a format the model can understand
    image = np.expand_dims(image, axis=0)

    #...get prediction
    predictions = model.predict(image, verbose=0)

    predicted_class = np.argmax(predictions[0])
    prediction_history.append(predicted_class)

    # after 1.0 seconds, choose the most common prediction
    if time.time() - start_time >= prediction_duration:

        most_common_class = Counter(prediction_history).most_common(1)[0][0]

        confidence = (
            prediction_history.count(most_common_class)
            / len(prediction_history)
        ) * 100

        label = f"{class_names[most_common_class]}: {confidence:.1f}%"

        # then reset for the next prediction
        prediction_history = []
        start_time = time.time()

    cv2.putText(
        frame,
        label,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("CSK Vision - Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
