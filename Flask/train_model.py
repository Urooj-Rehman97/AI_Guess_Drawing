import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# =========================
# SETTINGS
# =========================

categories = [
    "star",
    "moon",
    "apple",
    "ambulance",
    "basket",
    "banana",
    "bat",
    "bee",
    "book"
]

samples_per_category = 2000

IMG_SIZE = 28

# =========================
# LOAD DATA
# =========================

X = []
y = []

print("Loading dataset...\n")

for label, category in enumerate(categories):

    path = f"dataset/{category}.npy"

    data = np.load(path)

    print(
        category,
        "original:",
        data.shape
    )

    # Take samples
    data = data[:samples_per_category]

    X.append(data)

    # Numeric labels
    y.extend(
        [label] * len(data)
    )


# =========================
# COMBINE
# =========================

X = np.concatenate(X)

y = np.array(y)

print("\nFinal X:", X.shape)
print("Final y:", y.shape)


# =========================
# NORMALIZE
# =========================

X = X.astype("float32") / 255.0


# =========================
# RESHAPE FOR CNN
# =========================

# CNN expects:
# samples, height, width, channels

X = X.reshape(
    -1,
    28,
    28,
    1
)

print("CNN X shape:", X.shape)


# =========================
# TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining:", X_train.shape)
print("Testing:", X_test.shape)


# =========================
# DATA AUGMENTATION
# =========================

data_augmentation = tf.keras.Sequential([

    layers.RandomRotation(0.08),

  layers.RandomTranslation(
    height_factor=0.10,
    width_factor=0.10
),

    layers.RandomZoom(0.10)

])


# =========================
# CNN MODEL
# =========================

model = models.Sequential([

    layers.Input(
        shape=(28, 28, 1)
    ),

    # Augmentation
    data_augmentation,

    # ---------------------
    # CNN BLOCK 1
    # ---------------------

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # ---------------------
    # CNN BLOCK 2
    # ---------------------

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # ---------------------
    # CNN BLOCK 3
    # ---------------------

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # ---------------------
    # CLASSIFIER
    # ---------------------

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.4),

    layers.Dense(
        len(categories),
        activation="softmax"
    )

])


# =========================
# COMPILE
# =========================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# =========================
# MODEL SUMMARY
# =========================

model.summary()


# =========================
# CALLBACKS
# =========================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001
    )

]


# =========================
# TRAIN
# =========================

print("\nTraining CNN...\n")

history = model.fit(

    X_train,
    y_train,

    validation_split=0.1,

    epochs=30,

    batch_size=128,

    callbacks=callbacks,

    verbose=1

)


# =========================
# TEST
# =========================

print("\nEvaluating model...\n")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# =========================
# CLASSIFICATION REPORT
# =========================

y_pred = model.predict(
    X_test,
    verbose=0
)

y_pred_classes = np.argmax(
    y_pred,
    axis=1
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred_classes,
        target_names=categories
    )
)


# =========================
# SAVE MODEL
# =========================

model.save(
    "drawing_cnn.keras"
)

print(
    "\nModel saved as drawing_cnn.keras"
)