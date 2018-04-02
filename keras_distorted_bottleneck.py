# This file is for generating, caching and getting bottlenecks of model
import json
import random
import numpy as np
import os.path
from keras.preprocessing import image
import argparse
import sys
import re
from keras.utils import Sequence

# from model_source.deep_learning_models.inception_v3 import InceptionV3, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
# from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras_inception_bottleneck import get_image_path, create_image_lists
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
IM_WIDTH, IM_HEIGHT = 224, 224  # fixed size for InceptionV3
# IM_WIDTH, IM_HEIGHT = 299, 299  # fixed size for InceptionV3
ARGS = None
ROTATION_RANGE = 15
WIDTH_SHIFT_RANGE = 0.15
BRIGHTNESS_RANGE = (0.01, 1.3)
SHEAR_RANGE = 22
CHANNEL_SHIFT_RANGE = 20
HORIZONTAL_FLIP = True


def gen_base_model():
    # sub_base_model = ResNet50(weights='imagenet', include_top=False)
    sub_base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    x = sub_base_model.output
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=sub_base_model.input, outputs=x)


def predict_from_img(model, img):
    """Run model prediction on image
    Args:
      model: keras model
      img: image.load_img(img_file)
    Returns:
      list of predicted labels and their probabilities
    """
    if type(img) is list:
        x = []
        for i in img:
            x.append(image.img_to_array(i))
        x = preprocess_input(np.array(x))
        x = model.predict(x, verbose=1)
        return x
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)[0]
    return preds


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.

    Args:
      dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def create_or_load_training_data(model_bottle_dir, image_dir, testing_percentage,
                                 validation_percentage):
    if os.path.exists(model_bottle_dir + '/image_lists.json'):
        with open(model_bottle_dir + '/image_lists.json', 'r') as f:
            image_lists = json.load(f)
    else:
        image_lists = create_image_lists(image_dir, testing_percentage,
                                         validation_percentage)
        n_classes = len(image_lists.keys())
        if n_classes == 0:
            print(
                'No valid folders of images found at ' + image_dir)
            raise RuntimeError
        if n_classes == 1:
            print('Only one valid folder of images found at ' +
                  image_dir +
                  ' - multiple classes are needed for classification.')
            raise RuntimeError
        ensure_dir_exists(model_bottle_dir)
        with open(model_bottle_dir + '/image_lists.json', 'w') as f:
            json.dump(image_lists, f)
    return image_lists


def distort_image(img, rotation_range, width_shift_range, brightness_range,
                  shear_range, channel_shift_range, horizontal_flip=True):
    """Input: img object. Output: distorted img object"""
    # DO NOT MODIFY img, BUT GENERATE NEW IMAGE
    img_array = image.img_to_array(img)
    x = image.ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        brightness_range=brightness_range,
        shear_range=shear_range,
        horizontal_flip=horizontal_flip,
        channel_shift_range=channel_shift_range,
    ).random_transform(img_array)
    return image.array_to_img(x)


def get_or_create_bottleneck(
        image_lists, label_name, image_index, image_dir, category, bottleneck_dir,
        bottle_func, distorted=False, architecture='inception_v3'):
    # label_lists = image_lists[label_name]
    # sub_dir = label_lists['dir']
    # sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    # ensure_dir_exists(sub_dir_path)
    target_size = (IM_WIDTH, IM_HEIGHT)
    image_file = get_image_path(
        image_lists, label_name, image_index, image_dir, category)
    bottle_file = get_image_path(
        image_lists, label_name, image_index, bottleneck_dir, category) +\
        '_' + architecture + '.npy'
    try:
        bottleneck_values = np.load(bottle_file)
        return bottleneck_values
    except Exception as e:
        print('Bottleneck not found, creating bottleneck...\n{}'.format(e))
        if not distorted:
            img = image.load_img(image_file, target_size=target_size)
            np.save(bottle_file, bottle_func(img))
        else:
            img = image.load_img(re.sub('_\d+.jpg', '', image_file), target_size=target_size)
            distorted_image = distort_image(
                img, ROTATION_RANGE, WIDTH_SHIFT_RANGE,
                BRIGHTNESS_RANGE, SHEAR_RANGE, CHANNEL_SHIFT_RANGE, HORIZONTAL_FLIP)
            np.save(bottle_file, bottle_func(distorted_image))
        bottleneck_values = np.load(bottle_file)
        return bottleneck_values


def get_cached_bottlenecks(image_lists, how_many, category,
                           bottleneck_dir, image_dir, bottle_func, sequence=False,
                           architecture='inception_v3'):
    """Retrieves bottleneck sequence or batch for cached images.

    If no distortions are being applied, this function can retrieve the cached
    bottleneck values directly from disk for images. It picks a random set of
    images from the specified category.
    """
    if sequence:
        (bottlenecks, ground_truth, _) = get_random_cached_bottlenecks(
            image_lists, -1, category, bottleneck_dir, image_dir, bottle_func,
            architecture)

        class Image_Sequence(Sequence):

            def __init__(self, x_set, y_set, batch_size):
                self.x, self.y = x_set, y_set
                self.batch_size = batch_size

            def __len__(self):
                return int(np.ceil(len(self.x) / float(self.batch_size)))

            def __getitem__(self, idx):
                batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
                return np.array(batch_x), np.array(batch_y)
        return Image_Sequence(bottlenecks, ground_truth, how_many)
    else:
        (bottlenecks, ground_truth, _) = get_random_cached_bottlenecks(
            image_lists, how_many, category, bottleneck_dir, image_dir, bottle_func,
            architecture)
        return (np.array(bottlenecks), np.array(ground_truth))


def get_random_cached_bottlenecks(image_lists, how_many, category,
                                  bottleneck_dir, image_dir, bottle_func,
                                  architecture='inception_v3'):
    """Retrieves bottleneck values for cached images.

    If no distortions are being applied, this function can retrieve the cached
    bottleneck values directly from disk for images. It picks a random set of
    images from the specified category.

    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        # Retrieve a random sample of bottlenecks.
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name, image_index,
                                        image_dir, category)
            bottleneck = get_or_create_bottleneck(
                image_lists, label_name, image_index, image_dir, category,
                bottleneck_dir, bottle_func, architecture)
            bottlenecks.append(bottleneck)
            # y = np.zeros(class_count)
            # y[label_index] = 1
            ground_truths.append(label_index)
            filenames.append(image_name)
    else:
        # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(
                    image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index,
                                            image_dir, category)
                bottleneck = get_or_create_bottleneck(
                    image_lists, label_name, image_index, image_dir, category,
                    bottleneck_dir, bottle_func, architecture)
                bottlenecks.append(bottleneck)
                # y = np.zeros(class_count)
                # y[label_index] = 1
                ground_truths.append(label_index)
                filenames.append(image_name)
    return np.array(bottlenecks), np.array(ground_truths), np.array(filenames)


def cache_distort_bottlenecks(image_lists, bottle_func,
                              architecture='inception_v3'):
    '''generat distorted bottlenecks from image in image_lists'''
    distorted_image_lists = {}
    target_size = (IM_WIDTH, IM_HEIGHT)

    class Count():
        def __init__(self):
            self.n_skipped = 0
            self.n_created = 0

        def __call__(self, count_type='create'):
            if count_type == 'skipped':
                self.n_skipped += 1
                if self.n_skipped % 100 == 0:
                    print('{} existing bottlenecks skipped.'.format(self.n_skipped))
            else:
                self.n_created += 1
                if self.n_created % 100 == 0:
                    print('{} bottlenecks created.'.format(self.n_created))

        def print_total(self):
            print('{} bottlenecks created in total.'.format(self.n_created))
            print('{} existing bottlenecks skipped in total.'.format(self.n_skipped))
    count = Count()
    # creat bottlenecks for every distorted imgs
    for label_index, label_name in enumerate(image_lists.keys()):
        print('Current label: {}'.format(label_name))
        training_images = []
        label_lists = image_lists[label_name]  # image_lists['label 0']
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]  # label_lists['training']
            print('Current category: {}'.format(category))
            for image_index, image_name in enumerate(category_list):
                # 0, 'foo.jpg'
                # if label_index > 0 or image_index > 5:
                # if random.randrange(1000) < 998:  # ### DEBUG ONLY ###
                    # continue
                image_file = get_image_path(image_lists, label_name, image_index,
                                            ARGS.image_dir, category)
                img = image.load_img(image_file, target_size=target_size)
                # save original bottlenecks
                label_bott_dir = os.path.join(
                    ARGS.model_bottle_dir, 'distorted_bottlenecks', label_lists['dir'])
                ensure_dir_exists(label_bott_dir)
                save_path = os.path.join(
                    label_bott_dir, image_name + '_' + architecture + '.npy')
                if os.path.exists(save_path):
                    count('skipped')
                else:
                    np.save(save_path, bottle_func(img))
                    count()
                # save distorted image to path randomly
                to_save = random.randrange(1000) < ARGS.save_image_perthousand
                if category in ['testing', 'validation']:
                    continue
                training_images.append(image_name)
                # distort training image for {times_per_image} times
                for i in range(ARGS.times_per_image):
                    label_bott_dir = os.path.join(
                        ARGS.model_bottle_dir, 'distorted_bottlenecks', label_lists['dir'])
                    ensure_dir_exists(label_bott_dir)
                    distorted_image_name = image_name + \
                        '_{}'.format(i) + '.jpg'
                    save_path = os.path.join(
                        label_bott_dir, distorted_image_name + '_' + architecture + '.npy')
                    training_images.append(distorted_image_name)
                    if os.path.exists(save_path):
                        count('skipped')
                        continue
                    distorted_image = distort_image(
                        img, ROTATION_RANGE, WIDTH_SHIFT_RANGE,
                        BRIGHTNESS_RANGE, SHEAR_RANGE, CHANNEL_SHIFT_RANGE, HORIZONTAL_FLIP)
                    # randomly save distorted images for check
                    if to_save:
                        if i == 0:
                            label_image_dir = ARGS.image_dir + '_distorted'
                            ensure_dir_exists(label_image_dir)
                            img.save(os.path.join(label_image_dir, image_name))
                        img_save_path = os.path.join(
                            label_image_dir, distorted_image_name)
                        distorted_image.save(img_save_path)
                    bottleneck = bottle_func(distorted_image)
                    count()
                    np.save(save_path, bottleneck)

        distorted_image_lists[label_name] = {
            'dir': label_lists['dir'],
            'training': training_images,
            'testing': image_lists[label_name]['testing'],
            'validation': image_lists[label_name]['validation']}
    count.print_total()
    return distorted_image_lists


def main():
    print('validation percent: {}\n test percent: {}'.format(
        ARGS.validation_percentage, ARGS.testing_percentage))
    print('Obtaining image lists...')
    image_lists = create_or_load_training_data(
        ARGS.model_bottle_dir, ARGS.image_dir,
        ARGS.testing_percentage, ARGS.validation_percentage)
    print('Obtaining base model...')
    base_model = gen_base_model()
    print('Doing distortion...')
    distorted_image_lists = cache_distort_bottlenecks(
        image_lists, lambda img: predict_from_img(base_model, img))
    with open(ARGS.model_bottle_dir + '/distorted_image_lists.json', 'w') as f:
        json.dump(distorted_image_lists, f)
    print('Bottlenecks saved.')


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument(
        "--image_dir",
        default=r'C:\NN\clothes_styles\warm_up_train_20180201\Images\skirt_length_labels')
    a.add_argument(
        "--model_bottle_dir", default=r'C:\tmp\InceptionResNet\warm_up_skirt_length')
    a.add_argument("--save_image_perthousand", default=5, type=int)
    a.add_argument("--times_per_image", default=0, type=int)
    a.add_argument('--testing_percentage', type=int, default=10)
    a.add_argument('--validation_percentage', type=int, default=10)
    a.add_argument("--rotation_range", default=ROTATION_RANGE, type=int)
    a.add_argument("--width_shift_range", default=WIDTH_SHIFT_RANGE, type=float)
    a.add_argument("--brightness_low", default=BRIGHTNESS_RANGE[0], type=float)
    a.add_argument("--brightness_high", default=BRIGHTNESS_RANGE[1], type=float)
    a.add_argument("--shear_range", default=SHEAR_RANGE, type=float)
    a.add_argument("--channel_shift_range", default=CHANNEL_SHIFT_RANGE, type=float)
    a.add_argument("--horizontal_flip", default=HORIZONTAL_FLIP, type=bool)
    ARGS = a.parse_args()
    if ARGS.image_dir is None:
        a.print_help()
        sys.exit(1)
    if (not os.path.exists(ARGS.image_dir)):
        print("Directories do not exist")
        sys.exit(1)

    ARGS.brightness_range = (ARGS.brightness_low, ARGS.brightness_high)
    (ROTATION_RANGE, WIDTH_SHIFT_RANGE, BRIGHTNESS_RANGE, SHEAR_RANGE,
        HORIZONTAL_FLIP, CHANNEL_SHIFT_RANGE) =\
        (ARGS.rotation_range, ARGS.width_shift_range,
         ARGS.brightness_range, ARGS.shear_range, ARGS.horizontal_flip,
         ARGS.channel_shift_range)
    main()
