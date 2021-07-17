import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


image_directory = "IMAGE_DIRECTORY_PATH"
metadata_directory = "METADATA_PATH"


df_metadata = pd.read_csv(metadata_directory)
df_metadata = df_metadata.iloc[:3000]  # If less computation power then work with less data.


SIZE = 224  # Transoformed image dimensions
num_classes = 25
X_dataset = []


for i in range(df_metadata.shape[0]):
    img = image.load_img(image_directory + df_metadata['Id'][i]+'.jpg', target_size=(SIZE, SIZE, 3))
    img = image.img_to_array(img)
    img = img/255
    X_dataset.append(img)

X = np.array(X_dataset)  # Converting list of image(3D) into numpy array.
y = np.array(df_metadata.drop(columns=["Id", "Genre"]))  # keeping the column used as label.


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(SIZE, SIZE, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training and validating
# Train on 2400 samples, validate on 600 samples
history = model.fit(X, y, epochs=1, validation_split=0.2, batch_size=16)


# Testing with a single image
img = image.load_img('TEST_IMAGE_PATH', target_size=(SIZE, SIZE, 3))
img = image.img_to_array(img)
img = img/255.
img = np.expand_dims(img, axis=0)

classes = np.array(df_metadata.columns[2:])
model_output = model.predict(img)
sorted_categories = np.argsort(model_output[0])[:-11:-1]
for i in range(10):
    print("{}".format(classes[sorted_categories[i]]) +
          " ({:.3})".format(model_output[0][sorted_categories[i]]))


# Visualization of metrics
# Loss

plt.style.use('ggplot')

loss = history.history['loss']
val_loss = history.history['val_loss']

n_epochs = len(history.history['loss'])
epoch = range(0, n_epochs)

plt.plot(epoch, loss, color='tab:blue', label="Training loss")
plt.plot(epoch, val_loss, color='tab:red', label="Validation loss")

plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


# Accuracy

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

n_epochs = len(history.history['accuracy'])
epoch = range(0, n_epochs)

plt.plot(epoch, accuracy, color="tab:blue", label="Training accuracy")
plt.plot(epoch, val_accuracy, color="tab:red", label="Validation accuracy")

plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
