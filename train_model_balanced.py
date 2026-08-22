import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = "PlantVillage-Dataset"

# ==========================================
# LOAD DATASET
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, name)

print("\nTotal classes:", num_classes)

# ==========================================
# NORMALIZE IMAGES
# ==========================================

def normalize(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels

train_ds = train_ds.map(
    normalize,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    normalize,
    num_parallel_calls=tf.data.AUTOTUNE
)

# ==========================================
# DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ==========================================
# MOBILE NET V2
# ==========================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# ==========================================
# BUILD MODEL
# ==========================================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# CALLBACKS
# ==========================================

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=0.00001
)

# ==========================================
# STAGE 1
# ==========================================

print("\n===================================")
print("STAGE 1")
print("===================================")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[
        early_stop,
        reduce_lr
    ]
)

# ==========================================
# FINE-TUNING
# ==========================================

print("\n===================================")
print("STAGE 2 - FINE TUNING")
print("===================================")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[
        early_stop,
        reduce_lr
    ]
)

# ==========================================
# SAVE NEW MODEL
# ==========================================

model.save(
    "model/crop_disease_model_balanced.keras"
)

# ==========================================
# SAVE LABELS
# ==========================================

with open(
    "model/labels_balanced.txt",
    "w"
) as f:

    for name in class_names:
        f.write(name + "\n")

print("\n===================================")
print("NEW MODEL SAVED!")
print("===================================")
print("Model:")
print("model/crop_disease_model_balanced.keras")

print("\nLabels:")
print("model/labels_balanced.txt")

print("\nTotal classes:", num_classes)