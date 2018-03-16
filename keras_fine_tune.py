import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import json
import numpy as np

# from keras import __version__
# from keras.models import model_from_json
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from model_source.deep_learning_models.inception_v3 import InceptionV3, preprocess_input
#        , decode_predictions
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop
# from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
# from keras import backend as K
from keras.layers import Input

from keras.preprocessing import image

from keras_inception_bottleneck import \
    create_image_lists, cache_bottlenecks, get_cached_bottlenecks

IM_WIDTH, IM_HEIGHT = 299, 299  # fixed size for InceptionV3
FC_SIZE = 2048
# BASE_MODEL_OUTPUT_LAYER_INDEX = 311
# BASE_MODEL_OUTPUT_LAYER_NAME = 'global_average_pooling2d_1'
FINE_TUNE_FINAL_LAYER_INDEX = 279
FINE_TUNE_FINAL_LAYER_NAME = 'mixed9'


def predict_from_img(model, img):
    """Run model prediction on image
    Args:
      model: keras model
      img: image.load_img(img_file)
    Returns:
      list of predicted labels and their probabilities
    """
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)[0]
    return preds


def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def gen_base_model():
    sub_base_model = InceptionV3(weights='imagenet', include_top=False)
    x = sub_base_model.output
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=sub_base_model.input, outputs=x)


def add_final_layer(inputs, outputs, n_classes):
    x = Dense(FC_SIZE, activation='relu')(outputs)
    x = Dense(FC_SIZE // 2, activation='relu')(x)
    x = Dense(FC_SIZE // 4, activation='relu')(x)
    base_predictions = Dense(n_classes, activation='softmax')(x)
    # kernel_initializer='truncated_normal', bias_initializer='truncated_normal',
    return Model(inputs=inputs, outputs=base_predictions)


def set_trainable_layers(trainable_layer_list, frozen_layer_list):
    for layer in frozen_layer_list:
        layer.trainable = False
    for layer in trainable_layer_list:
        layer.trainable = True


def create_or_load_training_data(args):
    if os.path.exists(args.model_dir + '/image_lists.json'):
        with open(args.model_dir + '/image_lists.json', 'r') as f:
            image_lists = json.load(f)
        n_classes = len(image_lists.keys())
    else:
        image_lists = create_image_lists(args.image_dir, args.testing_percentage,
                                         args.validation_percentage)
        n_classes = len(image_lists.keys())
        if n_classes == 0:
            print(
                'No valid folders of images found at ' + args.image_dir)
            raise RuntimeError
        if n_classes == 1:
            print('Only one valid folder of images found at ' +
                  args.image_dir +
                  ' - multiple classes are needed for classification.')
            raise RuntimeError
        with open(args.model_dir + '/image_lists.json', 'w') as f:
            json.dump(image_lists, f)
    return image_lists, n_classes


def main(args):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""

    # n_classes = len(glob.glob(args.image_dir + "/*/"))
    # print('n_class: {}'.format(n_classes))
    # nb_val_samples = get_nb_files(args.val_dir)
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)
    base_model = None

    # model_file = os.path.join(
    #     args.model_dir, 'retrain_incep_v3_model_config.json')
    # if not os.path.exists(model_file):
    # model = add_new_last_layer(base_model, n_classes)
    # with open(model_file, 'w') as f:
    #     f.write(model.to_json())
    # else:
    #     with open(model_file) as f:
    #         model = model_from_json(f.read())
    #         print('reloading model...')

    image_lists, n_classes = create_or_load_training_data(args)

    if args.transfer_learning:
        # use bottleneck, here the model must be identical to the original top layer
        # print(base_model.output.shape)
        retrain_input_tensor = Input(shape=(2048,))
        print(retrain_input_tensor, retrain_input_tensor.shape)
        retrain_model = add_final_layer(
            retrain_input_tensor, retrain_input_tensor, n_classes)
        check_point_file = os.path.join(
            args.model_dir, "retrain_weights.hdf5")
        if os.path.exists(check_point_file):
            print('loading checkpoint {}'.format(check_point_file))
            retrain_model.load_weights(check_point_file)

        bottleneck_dir = os.path.join(
            args.model_dir, 'bottleneck_retrain_keras/')

        def bottle_pred_func(file):
            target_size = (IM_WIDTH, IM_HEIGHT)
            img = image.load_img(file, target_size=target_size)
            return predict_from_img(base_model, img)
        if not os.path.exists(bottleneck_dir):
            base_model = gen_base_model()
            cache_bottlenecks(image_lists, args.image_dir,
                              bottleneck_dir, bottle_pred_func)
        # train_sequence = get_cached_bottlenecks(
        #     image_lists, args.batch_size, 'training', bottleneck_dir,
        #     args.image_dir, bottle_pred_func)
        # validation_data = get_cached_bottlenecks(
        #     image_lists, -1, 'validation', bottleneck_dir,
        #     args.image_dir, bottle_pred_func, sequence=False)
        # # args.model_dir, "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
        # history_tl = retrain_model.fit_generator(
        #     train_sequence,
        #     epochs=nb_epoch,
        #     steps_per_epoch=nb_train_samples // batch_size,
        #     validation_data=validation_data,
        #     validation_steps=nb_train_samples // batch_size // 5,
        #     class_weight='auto', callbacks=callbacks_list)

        # for i in range(nb_train_samples // batch_size * nb_epoch):
        #     print('step: {}'.format(i))
        (x, y) = get_cached_bottlenecks(
            image_lists, -1, 'training',
            bottleneck_dir, args.image_dir, bottle_pred_func, sequence=False)
        val = get_cached_bottlenecks(
            image_lists, args.val_batch_size, 'validation',
            bottleneck_dir, args.image_dir, bottle_pred_func, sequence=False)
        checkpoint = ModelCheckpoint(check_point_file, monitor='val_acc',
                                     verbose=1, save_best_only=True, mode='max',
                                     save_weights_only=True)
        # tb_callback = TensorBoard(
        #     log_dir=args.model_dir, write_graph=True)
        callbacks_list = [checkpoint]  # , tb_callback]
        retrain_model.compile(optimizer=RMSprop(lr=0.0001),
                              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history_tl = retrain_model.fit(
            x, y, validation_data=val, epochs=nb_epoch,
            batch_size=batch_size,
            # steps_per_epoch=nb_train_samples // batch_size,
            # validation_steps=nb_train_samples // batch_size,
            callbacks=callbacks_list)
        (val_x, val_y) = get_cached_bottlenecks(
            image_lists, -1, 'validation',
            bottleneck_dir, args.image_dir, bottle_pred_func, sequence=False)
        # retrain_model.test_on_batch(val_x, val_y)
        if not args.no_plot:
            plot_training(history_tl)

    raise RuntimeError
    model = add_final_layer(base_model.input, base_model.output, n_classes)

    if args.fine_tune:
        assert model.layers[FINE_TUNE_FINAL_LAYER_INDEX].name == FINE_TUNE_FINAL_LAYER_NAME
        set_trainable_layers(
            trainable_layer_list=model.layers[:
                                              FINE_TUNE_FINAL_LAYER_INDEX + 1],
            frozen_layer_list=model.layers[FINE_TUNE_FINAL_LAYER_INDEX + 1:])
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history_ft = model.fit_generator(
            (x, y),
            epochs=nb_epoch,
            class_weight='auto')
        if not args.no_plot:
            plot_training(history_ft)

    # model.save(os.path.join(args.model_dir, 'inceptionv3-ft.model'))


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
    a.add_argument("--nb_epoch", default=30)
    a.add_argument("--batch_size", default=150)
    a.add_argument("--val_batch_size", default=150)
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
