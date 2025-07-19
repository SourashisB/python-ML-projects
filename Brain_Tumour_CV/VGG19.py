import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.api.applications.vgg19 import VGG19, preprocess_input
from keras.api.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.api.models import Model
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau


# Set paths to your data directories
base_dir = 'PREPROCESSED_DATA'
train_dir = os.path.join(base_dir, 'TRAIN')
val_dir = os.path.join(base_dir, 'VAL')
test_dir = os.path.join(base_dir, 'TEST')

# Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30 

# Data generators with augmentation for training
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.5, 1.5],
    shear_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

RANDOM_SEED = 42

# Load images from directories
train_gen = train_datagen.flow_from_directory(
    train_dir,
    color_mode='rgb',
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='binary',
    seed=RANDOM_SEED
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    color_mode='rgb',
    target_size=IMAGE_SIZE,
    batch_size=16,
    class_mode='binary',
    seed=RANDOM_SEED
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Build the model with VGG19 base
base_model = VGG19(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + (3,))
base_model.trainable = False  # Freeze VGG19 layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=6, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_gen,
    steps_per_epoch= 50,
    validation_data=val_gen,
    validation_steps= 25,
    epochs=EPOCHS,
    callbacks=[es, rlr]
)
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_gen,
    steps_per_epoch=50,
    validation_data=val_gen,
    validation_steps=25,
    epochs=EPOCHS,
    callbacks=[es, rlr]
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_gen, steps=test_gen.samples // BATCH_SIZE)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Plot training history
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.show()