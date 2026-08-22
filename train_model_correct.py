import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
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
# MOBILE NET V2 PREPROCESSING
# ==========================================

def preprocess_images(images, labels):

    images = tf.cast(images, tf.float32)

    images = preprocess_input(images)

    return images, labels


train_ds = train_ds.map(
    preprocess_images,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    preprocess_images,
    num_parallel_calls=tf.data.AUTOTUNE
)


train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


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

model = models.Sequential([

    layers.Input(shape=(224, 224, 3)),

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.1),

    layers.RandomZoom(0.1),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        num_classes,
        activation="softmax"
    )

])


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

    min_lr=0.000001

)


# ==========================================
# STAGE 1
# ==========================================

print("\n===================================")
print("STAGE 1: TRAINING")
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
# STAGE 2 - FINE TUNING
# ==========================================

print("\n===================================")
print("STAGE 2: FINE-TUNING")
print("===================================")


base_model.trainable = True


# Freeze most MobileNetV2 layers

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
# SAVE MODEL
# ==========================================

model.save(
    "model/crop_disease_model_correct.keras"
)


# ==========================================
# SAVE LABELS
# ==========================================

with open(
    "model/labels_correct.txt",
    "w"
) as f:

    for name in class_names:

        f.write(name + "\n")


print("\n===================================")
print("CORRECT MODEL SAVED SUCCESSFULLY!")
print("===================================")

print("Model:")
print("model/crop_disease_model_correct.keras")

print("\nLabels:")
print("model/labels_correct.txt")

print("\nTotal classes:", num_classes)

print("===================================")