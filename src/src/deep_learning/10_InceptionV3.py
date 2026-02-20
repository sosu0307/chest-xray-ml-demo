# Imports for model building
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import BatchNormalization
from sklearn.utils import class_weight

# Imports for image transformations
from tensorflow.keras.layers import RandomZoom
from tensorflow.keras.layers import RandomRotation
from tensorflow.keras.layers import RandomBrightness
from tensorflow.keras.layers import RandomContrast

# Imports for using InceptionV3
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Import callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Import Keras utility for image dataset loading
from tensorflow.keras.utils import image_dataset_from_directory

# Imports for assessing results
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Load training and test data. Image size 299x299 is required by InceptionV3.
train_ds = image_dataset_from_directory(
    "/content/chest_x_rays/Isolated_lung_images_train_test/Train/",
    batch_size=32,
    image_size=(299, 299)
)

val_ds = image_dataset_from_directory(
    "/content/chest_x_rays/Isolated_lung_images_train_test/Test/",
    batch_size=32,
    image_size=(299, 299)
)

class_names = val_ds.class_names

train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

# Create weights

counts = {0: 5074, 1: 12629}

# Option A: Manual Calculation
total = counts[0] + counts[1]
weight_for_0 = (1 / counts[0]) * (total / 2.0)
weight_for_1 = (1 / counts[1]) * (total / 2.0)

class_weights = {0: 2, 1: 0.5}

print(f"Weights: {class_weights}")

# Create callbacks to prevent overfitting
early_stopping = EarlyStopping(patience=10, # Wait for 10 epochs before applying
                               min_delta=0.001, # If the loss function doesn't change by 1% after 5 epochs, either up or down, we stop
                               verbose=1, # Display the epoch at which training stops
                               mode='min',
                               monitor='val_loss',
                               restore_best_weights=True)

reduce_learning_rate = ReduceLROnPlateau(monitor="val_loss",
                                         patience=3, # If val_loss stagnates for 3 consecutive epochs based on the min_delta value
                                         min_delta=0.01,
                                         factor=0.2,  # Reduce the learning rate by a factor of 0.2
                                         cooldown=4,  # Wait 4 epochs before retrying
                                         verbose=1)

model_checkpoint = ModelCheckpoint('best_inception_model.keras', monitor='val_loss', save_best_only=True)

base_model = InceptionV3(weights='imagenet', include_top=False)

# Freeze the layers of InceptionV3
base_model.trainable = False

# Create model
inputs = Input(shape=(299, 299, 3))

# Build the model
x = base_model(inputs)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(rate=0.4)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.Precision(name='precision')])

history_stage_1 = model.fit(train_ds,
                           validation_data=val_ds,
                           epochs=10,
                           class_weight=class_weights,
                           callbacks=[early_stopping, model_checkpoint])

for layer in base_model.layers[249:]:
    layer.trainable = True

# Re-compile with a very low learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.Precision(name='precision')])

# Train again (fine-tuning)
history_stage_2 = model.fit(train_ds,
                          validation_data=val_ds,
                          epochs=30,
                          class_weight=class_weights,
                          callbacks=[early_stopping, model_checkpoint, reduce_learning_rate])

# Get true labels and predictions from the validation/test set
true_labels = []
pred_labels = []

for images, labels in val_ds:
    pred = model.predict(images, verbose=0)
    pred_labels.extend((pred > 0.5).astype(int))
    true_labels.extend(labels.numpy())

# Display the classification report
print(classification_report(true_labels, pred_labels, target_names=class_names))

# Create the Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)

# Plot
plt.figure(figsize=(6,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('COVID-19 Detection Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Plot loss and accuracy by epoch

# Stage 1
plt.figure(figsize=(12,4))

plt.subplot(121)
plt.plot(history_stage_1.history['loss'])
plt.plot(history_stage_1.history['val_loss'])
plt.title('Model loss by epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')

plt.subplot(122)
plt.plot(history_stage_1.history['accuracy'])
plt.plot(history_stage_1.history['val_accuracy'])
plt.title('Model accuracy by epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')

plt.suptitle('Stage 1 Plots')
plt.savefig('stage_1_plots.png')
plt.show()

# Stage 2
plt.figure(figsize=(12,4))

plt.subplot(121)
plt.plot(history_stage_2.history['loss'])
plt.plot(history_stage_2.history['val_loss'])
plt.title('Model loss by epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')

plt.subplot(122)
plt.plot(history_stage_2.history['accuracy'])
plt.plot(history_stage_2.history['val_accuracy'])
plt.title('Model accuracy by epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')

plt.suptitle('Stage 2 Plots')
plt.savefig('stage_2_plots.png')
plt.show()