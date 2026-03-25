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

# 2. CREATE THE CONVOLUTIONAL BASE (Updated with 512 Filters)
model = models.Sequential()

# Block 1 - 64 Filters
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.BatchNormalization()) 
model.add(layers.MaxPooling2D((2, 2)))

# Block 2 - 256 Filters
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

# Block 3 - 512 Filters (Maximum Capacity)
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())

# 3. ADD DENSE LAYERS ON TOP
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu')) 
model.add(layers.Dropout(0.5)) # Vital to prevent the model from misidentifying the cat
model.add(layers.Dense(10))

# 4. COMPILE AND TRAIN MODEL
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training for 50 epochs to allow the 512 filters to learn properly
history = model.fit(train_images, train_labels, epochs=50, 
                    validation_data=(test_images, test_labels))

# 5. SAVE THE MODEL
model.save('final_model.h5')

# 6. PREDICTION FOR YOUR 4 IMAGES
def load_image(filename):
    img = preprocessing.image.load_img(filename, target_size=(32, 32))
    img = preprocessing.image.img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img = img / 255.0
    return img

def run_example():
    sample_files = ['cat_capture_0.png', 'airplane.png', 'truck.png', 'bird.png']
    model_loaded = models.load_model('final_model.h5')
    
    print("\n--- PREDICTION RESULTS ---")
    for filename in sample_files:
        if os.path.exists(filename):
            img = load_image(filename)
            pred_list = model_loaded.predict(img, verbose=0)
            result = np.argmax(pred_list[0])
            print(f"FILE: {filename:15} | PREDICTION: {class_names[result].upper()}")
        else:
            print(f"FILE: {filename:15} | ERROR: Not found")

run_example()