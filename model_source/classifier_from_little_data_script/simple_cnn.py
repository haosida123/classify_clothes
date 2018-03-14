'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
```
'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import os
import glob
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = r'C:\NN\clothes_styles\warm_up_train_20180201\Images_split\skirt_length_labels\train'
validation_data_dir = r'C:\NN\clothes_styles\warm_up_train_20180201\Images_split\skirt_length_labels\val'


def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


nb_train_samples = get_nb_files(train_data_dir)
print('nb_train_samples = {}'.format(nb_train_samples))
nb_validation_samples = get_nb_files(validation_data_dir)
print('validation_data_dir = {}'.format(validation_data_dir))
epochs = 10
batch_size = 100
N_CATEGORY = 6
CHECKPOINT_FILE = 'simple_cnn_checkpoint-best.hdf5'

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(N_CATEGORY))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

if os.path.exists(CHECKPOINT_FILE):
    print('loading checkpoint {}'.format(CHECKPOINT_FILE))
    model.load_weights(CHECKPOINT_FILE)
checkpoint = ModelCheckpoint(CHECKPOINT_FILE, monitor='val_acc',
                             verbose=1, save_best_only=True, mode='max',
                             save_weights_only=True)
tb_callback = TensorBoard(write_images=True,
    log_dir='/train_logs_simple_cnn', write_graph=True)
callbacks_list = [checkpoint, tb_callback]
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=callbacks_list)

model.save('simple_cnn.model')
