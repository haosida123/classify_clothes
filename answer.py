import os
import json
import argparse
import csv

# from keras import __version__
# from keras.models import model_from_json
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
#        , decode_predictions
# from keras.models import Model
# from keras.callbacks import TensorBoard
# from keras import backend as K
from keras.layers import Input
from keras.preprocessing import image

# from keras.preprocessing import image

from keras_distorted_bottleneck import \
    predict_from_img, gen_base_model
from keras_fine_tune import \
    add_final_layer, BOTTLENECK_DIM


def make_model(base_model, final_layer_func, final_weights_file, n_classes):
    model = final_layer_func(base_model.input, base_model.output, n_classes)
    retrain_input_tensor = Input(shape=(BOTTLENECK_DIM,))
    retrain_model = final_layer_func(
        retrain_input_tensor, retrain_input_tensor, n_classes)
    retrain_model.load_weights(final_weights_file)
    for (i, layer) in enumerate(reversed(retrain_model.layers)):
        model.layers[-(i + 1)].set_weights(layer.get_weights())
    return model


def main(args):
    base_model = gen_base_model()
    models, image_lists, label_lists = {}, {}, {}
    with open('answer.csv', 'w') as answer_file:
        answer_writer = csv.writer(answer_file, lineterminator='\n')
        with open(args.csv_file, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for [jpg_file, label_category, label] in csv_reader:
                weights_file = os.path.join(
                    args.retrain_base_dir, label_category, args.weights)
                if os.path.exists(weights_file) and label == '?':
                    print('\ncategory label: ', label_category)
                    if label_category not in models:
                        image_lists_path = os.path.join(args.retrain_base_dir,
                                                        label_category, 'image_lists.json')
                        assert os.path.exists(image_lists_path)
                        with open(image_lists_path, 'r') as f:
                            image_lists[label_category] = json.load(f)
                            n_classes = len(image_lists[label_category].keys())
                            label_lists[label_category] = \
                                sorted(
                                    [i for i in image_lists[label_category].keys()])
                            if label_lists[label_category][0] == 'label 0':
                                # There are some data that never behave as 'label 0'
                                # and our model doesn't predict
                                # but we still need to write its prob as 0 later
                                need_label_0 = False
                            else:
                                need_label_0 = True
                        models[label_category] = make_model(
                            base_model, add_final_layer, weights_file, n_classes
                        )
                    # write answers, eg. '0.4365;0.5180;0.8741;0.2189;0.3938;0.0422'
                    img = image.load_img(os.path.join(args.rank_dir, jpg_file))
                    pred = predict_from_img(models[label_category], img)
                    pred_ind = pred.argmax(axis=-1)
                    print('prediction for {}\n\tis {}. '.format(
                        jpg_file.split('/')[-1], pred_ind + int(need_label_0)), end='')
                    print('label:', label_lists[label_category][pred_ind])
                    label_pred = [0 for _ in range(int(need_label_0))]
                    label_pred.extend(pred)
                    label = ';'.join(['{:.5f}'.format(prob) for prob in label_pred])
                    print('output:', label)
                answer_writer.writerow([jpg_file, label_category, label])


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument(
        "--csv_file", default=r'C:\NN\clothes_styles\z_rank\Tests\answer.csv')
    a.add_argument("--retrain_base_dir", default=r'C:\tmp\InceptionResNet')
    a.add_argument("--rank_dir", default=r'C:\NN\clothes_styles\z_rank',
                   help='Dir to the folder for tests: Image')
    # a.add_argument("--image_lists", default='image_lists.json')
    a.add_argument("--weights", default="retrain_weights.hdf5")

    args = a.parse_args()
    main(args)
