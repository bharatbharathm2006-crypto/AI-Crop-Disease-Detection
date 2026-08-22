import tensorflow as tf
import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =========================================
# LOAD TRAINED MODEL
# =========================================

model = tf.keras.models.load_model(
    "model/crop_disease_model_correct.keras"
)


# =========================================
# LOAD CLASS LABELS
# =========================================

with open("model/labels_correct.txt", "r") as f:
    labels = [line.strip() for line in f]


IMG_SIZE = (224, 224)


# =========================================
# PREDICT CROP DISEASE
# =========================================

def predict_disease(image_path):

    # Load image
    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    # Convert image to NumPy array
    img = tf.keras.utils.img_to_array(img)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # MobileNetV2 preprocessing
    img = preprocess_input(
        img.astype(np.float32)
    )

    # AI prediction
    prediction = model.predict(
        img,
        verbose=0
    )

    # Get predicted class
    class_index = np.argmax(prediction)

    # Get confidence
    confidence = float(
        np.max(prediction) * 100
    )

    # Reject uncertain predictions
    if confidence < 60:
        return "Unsupported or Uncertain Leaf", confidence

    return labels[class_index], confidence