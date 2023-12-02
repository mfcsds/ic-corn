import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Lokasi direktori data (sesuaikan dengan path Anda)
train_dir = './dataset'
validation_dir = './dataset'

# Preprocessing data
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),  # Sesuaikan dengan ukuran input yang Anda inginkan
        batch_size=2,
        class_mode='categorical')  # 'binary' untuk klasifikasi 2 kelas, gunakan 'categorical' untuk multi-kelas

# Flow validation images in batches
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=2,
        class_mode='categorical')


number_of_class = 3
# Membangun model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(number_of_class, activation='softmax')  # Gunakan 'softmax' untuk klasifikasi multi-kelas
])

# Kompilasi model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Latih model
history = model.fit(
      train_generator,
      steps_per_epoch=10,  # Sesuaikan dengan jumlah data Anda
      epochs=20,
      validation_data=validation_generator,
      validation_steps=10,  # Sesuaikan dengan jumlah data validasi Anda
      verbose=2)

# Simpan model
model.save('mdj.h5')
