import os
import sys
import argparse
import matplotlib.pyplot as plt
import json
# import numpy as np

# from keras import __version__
# from keras.models import model_from_json
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
#        , decode_predictions
from keras.models import Model
# from keras.callbacks import TensorBoard
# from keras import backend as K
from keras.layers import Input

# from keras.preprocessing import image

from keras_distorted_bottleneck import \
    predict_from_img, gen_base_model, get_cached_bottlenecks
from keras_fine_tune import add_final_layer, load_training_data

IM_WIDTH, IM_HEIGHT = 299, 299  # fixed size for InceptionV3
FC_SIZE = 2048
# BASE_MODEL_OUTPUT_LAYER_INDEX = 311
# BASE_MODEL_OUTPUT_LAYER_NAME = 'global_average_pooling2d_1'
FINE_TUNE_FINAL_LAYER_INDEX = 279
FINE_TUNE_FINAL_LAYER_NAME = 'mixed9'


def main(args):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""

    base_model = None

    image_lists = load_training_data(args)
    n_classes = len(image_lists.keys())
        weights_file = os.path.join(
            args.model_dir, args.retrain_weights)

    if args.make_new_model:
        # use bottleneck, here the model must be identical to the original top layer
        # print(base_model.output.shape)
        base_model = gen_base_model()
        model = add_final_layer(
            base_model.input, base_model.output, n_classes)
        bottleneck_dir = os.path.join(
            args.model_dir, 'distorted_bottlenecks/')
        (val_x, val_y) = get_cached_bottlenecks(
            image_lists, -1, 'test',
            bottleneck_dir, args.image_dir, None, sequence=False)
        model.test_on_batch(val_x, val_y)
    # model.save(os.path.join(args.model_dir, 'inceptionv3-ft.model'))
    if not args.not_use_bottlenecks:
        retrain_input_tensor = Input(shape=(2048,))
        model = add_final_layer(
            retrain_input_tensor, retrain_input_tensor, n_classes)
        if os.path.exists(weights_file):
            print('loading checkpoint {}'.format(weights_file))
            model.load_weights(weights_file)


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument(
        "--image_dir", default=r'C:\NN\clothes_styles\warm_up_train_20180201\Images\skirt_length_labels')
    a.add_argument(
        "--model_dir", default=r'C:\tmp\warm_up_skirt_length')
    a.add_argument("--image_lists", default='image_lists.json')
    a.add_argument("--weights", default="retrain_weights.hdf5")
    a.add_argument("--make_new_model", default=False, action='store_true')
    a.add_argument("--not_use_bottlenecks", default=False, action='store_true')
    # a.add_argument("--val_batch_size", default=200)
    a.add_argument("--no_plot", default=False, action='store_true')
    a.add_argument("--transfer_learning", default=True)
    a.add_argument("--fine_tune", default=False)
    a.add_argument('--testing_percentage', type=int, default=0)
    a.add_argument('--validation_percentage', type=int, default=10)

    args = a.parse_args()
    if args.image_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.image_dir)):
        print("directories do not exist")
        sys.exit(1)

    main(args)
