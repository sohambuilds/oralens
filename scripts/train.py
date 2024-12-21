import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


train_dir = 'dataset/TRAIN'
test_dir = 'dataset/TEST'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True, 
    brightness_range=[0.7, 1.3], 
    fill_mode='constant',
    channel_shift_range=0.1  
)


test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
img_size = (224, 224)

# generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)


print("Class indices:", train_generator.class_indices)

# base model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet'
)
# freeze all layers
base_model.trainable = True


for layer in base_model.layers[:-30]:  # Freeze first 80% of layers
    layer.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(2, activation='softmax')  
])


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)#lower for finetuning

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='accuracy',  
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='accuracy',  
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Train 
epochs = 20

history = model.fit(
    train_generator,
    epochs=epochs,
    callbacks=[checkpoint, reduce_lr],
    verbose=1
)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\nTest accuracy: {test_accuracy:.4f}")

predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

class_names = list(train_generator.class_indices.keys())
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

model.save('final_model.keras')