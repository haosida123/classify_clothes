import os
import sys
import argparse
import numpy as np

# from keras import __version__
# from keras.models import model_from_json
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
#        , decode_predictions
# from keras.models import Model
# from keras.callbacks import TensorBoard
# from keras import backend as K
from keras.layers import Input

from keras.preprocessing import image

from keras_distorted_bottleneck import \
    predict_from_img, gen_base_model,\
    get_random_cached_bottlenecks, get_image_path
from keras_fine_tune import \
    add_final_layer, load_training_data, compile_retrain_model

IM_WIDTH, IM_HEIGHT = 299, 299  # fixed size for InceptionV3
# FC_SIZE = 2048
# BASE_MODEL_OUTPUT_LAYER_INDEX = 311
# BASE_MODEL_OUTPUT_LAYER_NAME = 'global_average_pooling2d_1'
FINE_TUNE_FINAL_LAYER_INDEX = 279
FINE_TUNE_FINAL_LAYER_NAME = 'mixed9'
BOTTLENECK_DIM = 1536


def make_model(base_model, final_layer_func, final_weights_file, n_classes):
    model = final_layer_func(base_model.input, base_model.output, n_classes)
    retrain_input_tensor = Input(shape=(2048,))
    retrain_model = final_layer_func(
        retrain_input_tensor, retrain_input_tensor, n_classes)
    retrain_model.load_weights(final_weights_file)
    for (i, layer) in enumerate(reversed(retrain_model.layers)):
        # print(i, layer.name)
        # print('\n')
        # print(model.layers[-(i+1)], layer)
        # print(model.layers[-(i+1)].weights)
        # print(layer.weights)
        model.layers[-(i + 1)].set_weights(layer.get_weights())
    return model


def main(args):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""

    base_model = None

    image_lists = load_training_data(args)
    n_classes = len(image_lists.keys())
    weights_file = os.path.join(
        args.model_dir, args.weights)
    if not os.path.exists(weights_file):
        print('weights_file: {} not found'.format(weights_file))
        raise FileNotFoundError
    bottleneck_dir = os.path.join(
        args.model_dir, 'distorted_bottlenecks/')

    if args.make_new_model:
        # use bottleneck, here the model must be identical to the original top layer
        # print(base_model.output.shape)
        target_size = (IM_WIDTH, IM_HEIGHT)
        base_model = gen_base_model()
        model = make_model(base_model, add_final_layer, weights_file, n_classes)
        for category in ['testing', 'validation', 'training']:
            print('\ncategory:', category)
            img_list = []
            truths_list = []
            for label_index, label_name in enumerate(image_lists.keys()):
                for image_index, image_name in enumerate(
                        image_lists[label_name][category]):
                    image_name = get_image_path(image_lists, label_name, image_index,
                                                args.image_dir, category)
                    img_list.append(image.load_img(image_name, target_size=target_size))
                    truths_list.append(label_index)
            preds = predict_from_img(model, img_list)
            pred_classes = preds.argmax(axis=-1)
            ind = np.nonzero(np.array(pred_classes) != np.array(truths_list))
            print('truths:', np.array(truths_list)[ind])
            print('preds:', pred_classes[ind], preds[ind])
            # img = image.load_img(file, target_size=target_size)
    # model.save(os.path.join(args.model_dir, 'inceptionv3-ft.model'))
    else:
        retrain_input_tensor = Input(shape=(BOTTLENECK_DIM,))
        retrain_model = add_final_layer(
            retrain_input_tensor, retrain_input_tensor, n_classes)
        print('loading weights: {}'.format(weights_file))
        retrain_model.load_weights(weights_file)
        compile_retrain_model(retrain_model)
        base_model = None

        def bottle_pred_func(img):
            # target_size = (IM_WIDTH, IM_HEIGHT)
            # img = image.load_img(file, target_size=target_size)
            return predict_from_img(base_model, img)

        for category in ['testing', 'validation', 'training']:
            print('\n{} data:'.format(category))
            (test_x, test_y, files) = get_random_cached_bottlenecks(
                image_lists, -1, category,
                bottleneck_dir, args.image_dir, bottle_pred_func)
            print(retrain_model.test_on_batch(test_x, test_y))
            preds = retrain_model.predict_on_batch(np.array(test_x))
            pred_classes = preds.argmax(axis=-1)
            ind = np.nonzero(np.array(pred_classes) != np.array(test_y))
            print('truths:', test_y[ind])
            print('preds:', pred_classes[ind], preds[ind])
            # for file in files[ind]:
            #     print(file)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument(
        "--model_dir", default=r'C:\tmp\InceptionResNet\warm_up_skirt_length')
    a.add_argument("--image_lists", default='image_lists.json')
    a.add_argument("--weights", default="retrain_weights.hdf5")
    # a.add_argument("--val_batch_size", default=200)
    a.add_argument("--make_new_model", default=False, action='store_true')
    a.add_argument(
        "--image_dir",
        default=r'C:\NN\clothes_styles\warm_up_train_20180201\Images\skirt_length_labels')
    a.add_argument("--predict_list", default='None.csv')
    a.add_argument("--predict_image", default='None.jpg')

    args = a.parse_args()
    if args.image_dir is None:
        a.print_help()
        sys.exit(1)

    main(args)
