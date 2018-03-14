# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import hashlib
import os.path
import random
import re
import json
import numpy as np
# from keras import backend as tf
import tensorflow as tf

from keras.utils import Sequence
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile

FLAGS = None

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
      image_dir: String path to a folder containing subfolders of images.
      testing_percentage: Integer percentage of the images to reserve for tests.
      validation_percentage: Integer percentage of images reserved for validation.

    Returns:
      A dictionary containing an entry for each label subfolder, with images split
      into training, testing, and validation sets within each label.
    """
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            print('No files found')
            continue
        if len(file_list) < 20:
            print(
                'WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            print(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a way of
            # grouping photos that are close variations of each other. For example
            # this is used in the plant disease data set to group multiple pictures of
            # the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(
                str.encode(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


def get_image_path(image_lists, label_name, index, image_dir, category):
    """"Returns a path to an image for a label at the given index.

    Args:
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Int offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string of the subfolders containing the training
      images.
      category: Name string of set to pull images from - training, testing, or
      validation.

    Returns:
      File system path string to an image that meets the requested parameters.

    """
    if label_name not in image_lists:
        print('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        print('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        print('Label %s has no images in the category %s.',
              label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category, architecture):
    """"Returns a path to a bottleneck file for a label at the given index.

    Args:
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Integer offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      bottleneck_dir: Folder string holding cached files of bottleneck values.
      category: Name string of set to pull images from - training, testing, or
      validation.
      architecture: The name of the model architecture.

    Returns:
      File system path string to an image that meets the requested parameters.
    """
    return get_image_path(image_lists, label_name, index, bottleneck_dir,
                          category) + '_' + architecture + '.npy'


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.

    Args:
      dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


bottleneck_path_2_bottleneck_values = {}


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, bottle_func):
    """Create a single bottleneck file."""
    print('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index,
                                image_dir, category)
    if not gfile.Exists(image_path):
        print('File does not exist %s', image_path)
    try:
        bottleneck_values = bottle_func(image_path)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                     str(e)))
    np.save(bottleneck_path, bottleneck_values)


def get_or_create_bottleneck(image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, bottle_func,
                             architecture='inception_v3'):
    """Retrieves or calculates bottleneck values for an image.
    If a cached version of the bottleneck data exists on-disk, return that,
    otherwise calculate the data and save it to disk for future use.
    """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          bottleneck_dir, category, architecture)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, bottle_func)
    # with open(bottleneck_path, 'r') as bottleneck_file:
    #     bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        # with open(bottleneck_path, 'r') as f:
        bottleneck_values = np.load(bottleneck_path)
        # bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except Exception as e:
        print('Invalid float found, recreating bottleneck\n{}'.format(e))
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, bottle_func)
        # with open(bottleneck_path, 'r') as bottleneck_file:
        #     bottleneck_string = bottleneck_file.read()
        # with open(bottleneck_path, 'r') as f:
        bottleneck_values = np.load(bottleneck_path)
        # Allow exceptions to propagate here, since they shouldn't happen after a
        # fresh creation
        # bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def cache_bottlenecks(image_lists, image_dir, bottleneck_dir,
                      bottle_func):
    """Ensures all the training, testing, and validation bottlenecks are cached.

    Because we're likely to read the same image multiple times (if there are no
    distortions applied during training) it can speed things up a lot if we
    calculate the bottleneck layer values once for each image during
    preprocessing, and then just read those cached values repeatedly during
    training. Here we go through all the images we've found, calculate those
    values, and save them off.
    """
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(image_lists, label_name, index,
                                         image_dir, category, bottleneck_dir, bottle_func)

                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    print(
                        str(how_many_bottlenecks) + ' bottleneck files created.')


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


def cached_bottlenecks_sequence(image_lists, how_many, category,
                                bottleneck_dir, image_dir, bottle_func, sequence=True):
    """Retrieves bottleneck sequence for cached images.

    If no distortions are being applied, this function can retrieve the cached
    bottleneck values directly from disk for images. It picks a random set of
    images from the specified category.
    """
    (bottlenecks, ground_truth, _) = get_random_cached_bottlenecks(
        image_lists, -1, category, bottleneck_dir, image_dir, bottle_func)
    if sequence:
        return Image_Sequence(bottlenecks, ground_truth, how_many)
    else:
        return (np.array(bottlenecks), np.array(ground_truth))


def get_random_cached_bottlenecks(image_lists, how_many, category,
                                  bottleneck_dir, image_dir, bottle_func):
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
                bottleneck_dir, bottle_func)
            bottlenecks.append(bottleneck)
            y = np.zeros(class_count)
            y[label_index] = 1
            ground_truths.append(y)
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
                    bottleneck_dir, bottle_func)
                bottlenecks.append(bottleneck)
                y = np.zeros(class_count)
                y[label_index] = 1
                ground_truths.append(y)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames


def get_random_distorted_bottlenecks(
        sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
        distorted_image, resized_input_tensor, bottleneck_tensor):
    """Retrieves bottleneck values for training images, after distortions.

    If we're training with distortions like crops, scales, or flips, we have to
    recalculate the full model for every image, and so we can't use cached
    bottleneck values. Instead we find random images for the requested category,
    run them through the distortion graph, and then the full graph to get the
    bottleneck results for each.

    Args:
      sess: Current TensorFlow Session.
      image_lists: Dictionary of training images for each label.
      how_many: The integer number of bottleneck values to return.
      category: Name string of which set of images to fetch - training, testing,
      or validation.
      image_dir: Root folder string of the subfolders containing the training
      images.
      input_jpeg_tensor: The input layer we feed the image data to.
      distorted_image: The output node of the distortion graph.
      resized_input_tensor: The input node of the recognition graph.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.

    Returns:
      List of bottleneck arrays and their corresponding ground truths.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir,
                                    category)
        if not gfile.Exists(image_path):
            print('File does not exist %s', image_path)
        jpeg_data = gfile.FastGFile(image_path, 'rb').read()
        # Note that we materialize the distorted_image_data as a numpy array before
        # sending running inference on the image. This involves 2 memory copies and
        # might be optimized in other implementations.
        distorted_image_data = sess.run(distorted_image,
                                        {input_jpeg_tensor: jpeg_data})
        bottleneck_values = sess.run(bottleneck_tensor,
                                     {resized_input_tensor: distorted_image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        bottlenecks.append(bottleneck_values)
        ground_truths.append(label_index)
    return bottlenecks, ground_truths


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
    """Whether any distortions are enabled, from the input flags.

    Args:
      flip_left_right: Boolean whether to randomly mirror images horizontally.
      random_crop: Integer percentage setting the total margin used around the
      crop box.
      random_scale: Integer percentage of how much to vary the scale by.
      random_brightness: Integer range to randomly multiply the pixel values by.

    Returns:
      Boolean value indicating whether any distortions should be applied.
    """
    return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
            (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness, input_width, input_height,
                          input_depth, input_mean, input_std):
    """Creates the operations to apply the specified distortions.

    During training it can help to improve the results if we run the images
    through simple distortions like crops, scales, and flips. These reflect the
    kind of variations we expect in the real world, and so can help train the
    model to cope with natural data more effectively. Here we take the supplied
    parameters and construct a network of operations to apply them to an image.

    Cropping
    ~~~~~~~~

    Cropping is done by placing a bounding box at a random position in the full
    image. The cropping parameter controls the size of that box relative to the
    input image. If it's zero, then the box is the same size as the input and no
    cropping is performed. If the value is 50%, then the crop box will be half the
    width and height of the input. In a diagram it looks like this:

    <       width         >
    +---------------------+
    |                     |
    |   width - crop%     |
    |    <      >         |
    |    +------+         |
    |    |      |         |
    |    |      |         |
    |    |      |         |
    |    +------+         |
    |                     |
    |                     |
    +---------------------+

    Scaling
    ~~~~~~~

    Scaling is a lot like cropping, except that the bounding box is always
    centered and its size varies randomly within the given range. For example if
    the scale percentage is zero, then the bounding box is the same size as the
    input and no scaling is applied. If it's 50%, then the bounding box will be in
    a random range between half the width and height and full size.

    Args:
      flip_left_right: Boolean whether to randomly mirror images horizontally.
      random_crop: Integer percentage setting the total margin used around the
      crop box.
      random_scale: Integer percentage of how much to vary the scale by.
      random_brightness: Integer range to randomly multiply the pixel values by.
      graph.
      input_width: Horizontal size of expected input image to model.
      input_height: Vertical size of expected input image to model.
      input_depth: How many channels the expected input image should have.
      input_mean: Pixel value that should be zero in the image for the graph.
      input_std: How much to divide the pixel values by before recognition.

    Returns:
      The jpeg input layer and the distorted result tensor.
    """

    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                           minval=1.0,
                                           maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, input_width)
    precrop_height = tf.multiply(scale_value, input_height)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    cropped_image = tf.random_crop(precropped_image_3d,
                                   [input_height, input_width, input_depth])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                         minval=brightness_min,
                                         maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    offset_image = tf.subtract(brightened_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')
    return jpeg_data, distort_result


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.

    Args:
      result_tensor: The new final node that produces results.
      ground_truth_tensor: The node we feed ground truth data
      into.

    Returns:
      Tuple of (evaluation step, prediction).
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, ground_truth_tensor)
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def add_jpeg_decoding(input_width, input_height, input_depth, input_mean,
                      input_std):
    """Adds operations that perform JPEG decoding and resizing to the graph..

    Args:
      input_width: Desired width of the image fed into the recognizer graph.
      input_height: Desired width of the image fed into the recognizer graph.
      input_depth: Desired channels of the image fed into the recognizer graph.
      input_mean: Pixel value that should be zero in the image for the graph.
      input_std: How much to divide the pixel values by before recognition.

    Returns:
      Tensors for the node to feed JPEG data into, and the output of the
        preprocessing steps.
    """
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return jpeg_data, mul_image


def main(_):

    # Prepare necessary directories that can be used during training
    # prepare_file_system()

    meta_graph = FLAGS.ckpt_dir + '/new_graph.meta'
    if os.path.exists(meta_graph):
        print('restarting graph...')
        restart_graph = True
        new_saver = tf.train.import_meta_graph(meta_graph)
        graph = tf.get_default_graph()
    else:
        print('creating new graph...')
        restart_graph = False
        # Gather information about the model architecture we'll be using.
        model_info = 'create_model_info(FLAGS.architecture)'
        if not model_info:
            print('Did not recognize architecture flag')
            return -1

        # Set up the pre-trained graph.
        # maybe_download_and_extract(model_info['data_url'])
        graph, bottleneck_tensor, resized_image_tensor = "(\
            create_model_graph(model_info))"

    # Look at the folder structure, and create lists of all the images.
    if os.path.exists(FLAGS.ckpt_dir + '/image_lists.json'):
        with open(FLAGS.ckpt_dir + '/image_lists.json', 'r') as f:
            image_lists = json.load(f)
        class_count = len(image_lists.keys())
    else:
        image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                         FLAGS.validation_percentage)
        class_count = len(image_lists.keys())
        if class_count == 0:
            print(
                'No valid folders of images found at ' + FLAGS.image_dir)
            return -1
        if class_count == 1:
            print('Only one valid folder of images found at ' +
                  FLAGS.image_dir +
                  ' - multiple classes are needed for classification.')
            return -1
        with open(FLAGS.ckpt_dir + '/image_lists.json', 'w') as f:
            json.dump(image_lists, f)

    # See if the command-line flags mean we're applying any distortions.
    do_distort_images = should_distort_images(
        FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
        FLAGS.random_brightness)

    with tf.Session(graph=graph) as sess:
        if restart_graph is False:
            # Set up the image decoding sub-graph.
            jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
                model_info['input_width'], model_info['input_height'],
                model_info['input_depth'], model_info['input_mean'],
                model_info['input_std'])
        else:
            # load saved graph if there is
            ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
            if not (ckpt and ckpt.model_checkpoint_path):
                init = tf.global_variables_initializer()
                sess.run(init)
            else:
                new_saver.restore(sess, ckpt.model_checkpoint_path)
            bottleneck_tensor = graph.get_tensor_by_name(
                'pool_3/_reshape:0')
            resized_image_tensor = graph.get_tensor_by_name('Mul:0')
            jpeg_data_tensor = graph.get_tensor_by_name('DecodeJPGInput:0')
            decoded_image_tensor = graph.get_tensor_by_name('Mul_1:0')
            train_step = graph.get_operation_by_name(
                'train/GradientDescent')
            evaluation_step = graph.get_tensor_by_name(
                'accuracy/accuracy/Mean:0')
            prediction = graph.get_tensor_by_name(
                'accuracy/correct_prediction/ArgMax:0')
            cross_entropy = graph.get_tensor_by_name(
                'cross_entropy/sparse_softmax_cross_entropy_loss/value:0')
            bottleneck_input = graph.get_tensor_by_name(
                'input/BottleneckInputPlaceholder:0')
            ground_truth_input = graph.get_tensor_by_name(
                'input/GroundTruthInput:0')
            final_tensor = graph.get_tensor_by_name('final_result:0')
            merged = graph.get_tensor_by_name('Merge/MergeSummary:0')
            counter = graph.get_tensor_by_name('step_counter:0')
            learning_rate = graph.get_tensor_by_name('learning_rate:0')

        if do_distort_images:
            # We will be applying distortions, so setup the operations we'll need.
            (distorted_jpeg_data_tensor,
             distorted_image_tensor) = add_input_distortions(
                 FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
                 FLAGS.random_brightness, model_info['input_width'],
                 model_info['input_height'], model_info['input_depth'],
                 model_info['input_mean'], model_info['input_std'])
        elif not os.path.exists(FLAGS.ckpt_dir + '/image_lists.json'):
            # We'll make sure we've calculated the 'bottleneck' image summaries and
            # cached them on disk.
            cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                              FLAGS.bottleneck_dir, jpeg_data_tensor,
                              decoded_image_tensor, resized_image_tensor,
                              bottleneck_tensor, FLAGS.architecture)

        if restart_graph is False:
            # Add the new layer that we'll be training.
            global_step = tf.Variable(-1, trainable=False, name='global_step')
            counter = tf.assign_add(global_step, 1, name='step_counter')
            learning_rate = tf.placeholder(
                tf.float32, shape=[], name='learning_rate')
            (train_step, cross_entropy, bottleneck_input, ground_truth_input,
             final_tensor) = '''add_final_training_ops(
                 global_step, len(image_lists.keys()),
                 FLAGS.final_tensor_name, bottleneck_tensor,
                 model_info['bottleneck_tensor_size'], learning_rate)'''

            # Create the operations we need to evaluate the accuracy of our new layer.
            evaluation_step, prediction = add_evaluation_step(
                final_tensor, ground_truth_input)

            # Merge all the summaries and write them out to the summaries_dir
            merged = tf.summary.merge_all()
            # Set up all our weights to their initial default values.
            init = tf.global_variables_initializer()
            sess.run(init)
            new_saver = tf.train.Saver()
            tf.train.export_meta_graph(
                filename=FLAGS.ckpt_dir + '/new_graph.meta')
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                             sess.graph)
        validation_writer = tf.summary.FileWriter(
            FLAGS.summaries_dir + '/validation')

        # Run the training for as many cycles as requested on the command line.
        for j in range(FLAGS.training_steps):
            i = sess.run(counter)
            # Get a batch of input bottleneck values, either calculated fresh every
            # time with distortions applied, or from the cache stored on disk.
            if do_distort_images:
                (train_bottlenecks,
                 train_ground_truth) = get_random_distorted_bottlenecks(
                     sess, image_lists, FLAGS.train_batch_size, 'training',
                     FLAGS.image_dir, distorted_jpeg_data_tensor,
                     distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
            else:
                (train_bottlenecks,
                 train_ground_truth, _) = get_random_cached_bottlenecks(
                     sess, image_lists, FLAGS.train_batch_size, 'training',
                     FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                     decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                     FLAGS.architecture)
            feed_dict = {bottleneck_input: train_bottlenecks,
                         ground_truth_input: train_ground_truth,
                         learning_rate: FLAGS.learning_rate}
            # Feed the bottlenecks and ground truth into the graph, and run a training
            # step. Capture training summaries for TensorBoard with the `merged` op.
            train_summary, _ = sess.run(
                [merged, train_step],
                feed_dict=feed_dict)
            train_writer.add_summary(train_summary, i)

            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == FLAGS.training_steps)
            if (i % (10 * FLAGS.eval_step_interval)) == 0 or is_last_step:
                new_saver.save(sess, FLAGS.ckpt_dir + '/checkpoint',
                               global_step=i, write_meta_graph=False)
                print('%s: Step %d: ' % (datetime.now(), i))
                print('checkpoint saved.')
            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run(
                    [evaluation_step, cross_entropy],
                    feed_dict=feed_dict)
                validation_bottlenecks, validation_ground_truth, _ = (
                    get_random_cached_bottlenecks(
                        sess, image_lists, FLAGS.validation_batch_size, 'validation',
                        FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                        FLAGS.architecture))
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                validation_summary, validation_accuracy = sess.run(
                    [merged, evaluation_step],
                    feed_dict=feed_dict)
                validation_writer.add_summary(validation_summary, i)
                print('Tra acc = %.1f%%   Cro ent = %f   Val acc = %.1f%%' %
                      (train_accuracy * 100, cross_entropy_value, validation_accuracy * 100))

            # Store intermediate results
            intermediate_frequency = FLAGS.intermediate_store_frequency

            if (intermediate_frequency > 0 and
                    (i % intermediate_frequency == 0) and i > 0):
                intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
                                          'intermediate_' + str(i) + '.pb')
                print('Save intermediate result to : ' +
                      intermediate_file_name)
                # save_graph_to_file(sess, graph, intermediate_file_name)

        # We've completed all our training, so run a final test evaluation on
        # some new images we haven't used before.
        test_bottlenecks, test_ground_truth, test_filenames = (
            get_random_cached_bottlenecks(
                sess, image_lists, FLAGS.test_batch_size, 'testing',
                FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                FLAGS.architecture))
        test_accuracy, predictions = sess.run(
            [evaluation_step, prediction],
            feed_dict={bottleneck_input: test_bottlenecks,
                       ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%% (N=%d)' %
              (test_accuracy * 100, len(test_bottlenecks)))

        if FLAGS.print_misclassified_test_images:
            print('=== MISCLASSIFIED TEST IMAGES ===')
            for i, test_filename in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i]:
                    print('%70s  %s' %
                          (test_filename,
                           list(image_lists.keys())[predictions[i]]))

        # # Write out the trained graph and labels with the weights stored as
        # # constants.
        # save_graph_to_file(sess, graph, FLAGS.output_graph)
        # with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
        #     f.write('\n'.join(image_lists.keys()) + '\n')

        # export_model(sess, FLAGS.architecture, FLAGS.saved_model_dir)
