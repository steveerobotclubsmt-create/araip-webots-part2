import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. DATA PREPARATION
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Preview Grid (From cnn_v3)
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([]); plt.yticks([]); plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# 2. CREATE THE MODEL (High capacity + Data Augmentation)
# We add a RandomFlip/Rotation layer to help the model stop
# obsessing over the background "sky" position.
model = models.Sequential([
    # Data Augmentation (Helps fix the "Airplane" bias)
    layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),
    layers.RandomRotation(0.1),

    # Block 1 - 64 Filters
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 2 - 256 Filters
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 3 - 512 Filters
    layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Dense Head
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5), # Crucial for preventing "Airplane" overconfidence
    layers.Dense(10)
])

# --- THE TABLE (Shows before training) ---
print("\n--- MODEL ARCHITECTURE TABLE ---")
model.summary()

# 3. COMPILE AND TRAIN
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training for 50 epochs
history = model.fit(train_images, train_labels, epochs=50, batch_size=128,
                    validation_data=(test_images, test_labels))

# --- PLOT TRAINING HISTORY ---
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.ylim([0, 1]); plt.legend(loc='lower right')
plt.show()

# 4. PREDICTION FOR 4 IMAGES IN 1 RUN
def run_example():
    # Update these filenames to the 4 pictures you want to detect
    sample_files = ['cat_capture_0.png', 'airplane.png', 'truck.png', 'bird.png']

    print("\n--- STARTING 4-IMAGE DETECTION ---")
    for filename in sample_files:
        if os.path.exists(filename):
            # Load and process image
            img = preprocessing.image.load_img(filename, target_size=(32, 32))
            img_arr = preprocessing.image.img_to_array(img)
            img_arr = img_arr.reshape(1, 32, 32, 3) / 255.0

            # Predict
            pred = model.predict(img_arr, verbose=0)
            prob = tf.nn.softmax(pred[0]).numpy()
            idx = np.argmax(prob)
            conf = prob[idx] * 100

            # PUBLISH RESULTS (Text first, then picture)
            print(f"File: {filename} --> AI Prediction: {class_names[idx].upper()} ({conf:.2f}%)")

            plt.figure(figsize=(3,3))
            plt.imshow(img)
            plt.title(f"AI Guessed: {class_names[idx]} ({conf:.2f}%)")
            plt.axis('off')
            plt.show()
        else:
            print(f"File: {filename} --> ERROR: Not found")

run_example()