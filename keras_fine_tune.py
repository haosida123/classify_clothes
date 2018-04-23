import os
# import sys
import glob
import argparse
import matplotlib.pyplot as plt
import json
import subprocess
# import numpy as np

# from keras import __version__
# from keras.models import model_from_json
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
#        , decode_predictions
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, RMSprop
# from keras import regularizers
from keras.models import Model
# from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping
# from keras import backend as K
from keras.layers import Input

# from keras.preprocessing import image

from keras_distorted_bottleneck import \
    cache_distort_bottlenecks, get_cached_bottlenecks, predict_from_img, gen_base_model

IM_WIDTH, IM_HEIGHT = 299, 299  # fixed size for InceptionV3
# IM_WIDTH, IM_HEIGHT = 224, 224
# BASE_MODEL_OUTPUT_LAYER_INDEX = 311
# BASE_MODEL_OUTPUT_LAYER_NAME = 'global_average_pooling2d_1'
FINE_TUNE_FINAL_LAYER_INDEX = 279
FINE_TUNE_FINAL_LAYER_NAME = 'mixed9'
BOTTLENECK_DIM = 1536
# BOTTLENECK_DIM = 2048


def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def add_final_layer(inputs, outputs, n_classes):
    # x = Dropout(0.02)(outputs)
    x = Dense(128, activation='relu')(outputs)
    #           activity_regularizer=regularizers.l2(0.00001))(x)
    # x = Dropout(0.2)(x)
    # x = Dense(32, activation='relu')(x)
    # x = Dropout(0.02)(x)
    x = Dense(12, activation='relu')(x)
    # x = Dropout(0.01)(x)
    base_predictions = Dense(n_classes, activation='softmax')(x)
    # kernel_initializer='truncated_normal', bias_initializer='truncated_normal',
    return Model(inputs=inputs, outputs=base_predictions)


def add_final_layer_best(inputs, outputs, n_classes):
    x = Dropout(0.1)(outputs)
    x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.05)(x)
    x = Dense(512, activation='relu')(x)
    # x = Dropout(0.05)(x)
    base_predictions = Dense(n_classes, activation='softmax')(x)
    # kernel_initializer='truncated_normal', bias_initializer='truncated_normal',
    return Model(inputs=inputs, outputs=base_predictions)


def compile_retrain_model(retrain_model, learning_rate=0.00003):
    retrain_model.compile(optimizer=RMSprop(lr=learning_rate),
                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def set_trainable_layers(trainable_layer_list, frozen_layer_list):
    for layer in frozen_layer_list:
        layer.trainable = False
    for layer in trainable_layer_list:
        layer.trainable = True


def load_training_data(args):
    ''' Need: Sub directory: args.model_dir and Json file: args.image_lists
    to return loaded image_lists '''
    image_lists_path = os.path.join(args.model_dir, args.image_lists)
    if os.path.exists(image_lists_path):
        with open(image_lists_path, 'r') as f:
            image_lists = json.load(f)
    else:
        print(
            '...File not exist, please create bottlenecks using distorted_bottleneck.py first.')
        print(args.image_lists)
    n_train, n_val, n_test = 0, 0, 0
    for label in image_lists.keys():
        n_train += len(image_lists[label]['training'])
        n_val += len(image_lists[label]['validation'])
        n_test += len(image_lists[label]['testing'])
    print('...{}: train: {} samples, val: {} samples, test: {} samples.'.format(
        args.image_lists, n_train, n_val, n_test
    ))
    return image_lists


def main(args):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""

    # n_classes = len(glob.glob(args.image_dir + "/*/"))
    # print('n_class: {}'.format(n_classes))
    # nb_val_samples = get_nb_files(args.val_dir)
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)
    base_model = None

    image_lists = load_training_data(args)
    n_classes = len(image_lists.keys())

    if args.transfer_learning:
        # use bottleneck, here the model must be identical to the original top layer
        # print(base_model.output.shape)
        retrain_input_tensor = Input(shape=(BOTTLENECK_DIM,))
        print('...making transfer layers')
        retrain_model = add_final_layer(
            retrain_input_tensor, retrain_input_tensor, n_classes)
        check_point_file = os.path.join(
            args.model_dir, args.retrain_weights)
        if os.path.exists(check_point_file):
            print('...loading checkpoint {}'.format(check_point_file))
            try:
                retrain_model.load_weights(check_point_file)
            except Exception as e:
                print('...checkpoint loading failed, write new checkpoint')

        bottleneck_dir = os.path.join(
            args.model_dir, 'distorted_bottlenecks/')

        def bottle_pred_func(img):
            # target_size = (IM_WIDTH, IM_HEIGHT)
            # img = image.load_img(file, target_size=target_size)
            return predict_from_img(base_model, img)
        if not os.path.exists(bottleneck_dir):
            print('...bottlenecks not found...')
            base_model = gen_base_model()
            cache_distort_bottlenecks(image_lists, args.image_dir,
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
        print('...generating training and validation set')
        (x, y) = get_cached_bottlenecks(
            image_lists, -1, 'training',
            bottleneck_dir, args.image_dir, bottle_pred_func, sequence=False)
        val = get_cached_bottlenecks(
            image_lists, -1, 'validation',
            bottleneck_dir, args.image_dir, bottle_pred_func, sequence=False)
        checkpoint = ModelCheckpoint(check_point_file,
                                     verbose=0, save_best_only=True,
                                     save_weights_only=True)
        earlystopping = EarlyStopping(patience=args.earlystopping_patience,
                                      monitor='val_loss')
        # tb_callback = TensorBoard(
        #     log_dir=args.model_dir, write_graph=True)
        callbacks_list = [checkpoint, earlystopping]  # , tb_callback]
        # use compile function to compile optimizer, loss func and metrics
        compile_retrain_model(retrain_model, args.learning_rate)
        print('...begin training')
        history_tl = retrain_model.fit(
            x, y, validation_data=val, epochs=nb_epoch,
            batch_size=batch_size,
            verbose=0,
            # steps_per_epoch=nb_train_samples // batch_size,
            # validation_steps=nb_train_samples // batch_size,
            callbacks=callbacks_list).history
        retrain_model.load_weights(check_point_file)
        (test_x, test_y) = get_cached_bottlenecks(
            image_lists, -1, 'testing',
            bottleneck_dir, args.image_dir, bottle_pred_func, sequence=False)
        test = retrain_model.test_on_batch(test_x, test_y)
        print('\n...test loss: {0:.3f}, tess acc: {1:.3f}\n'.format(*test))
        history_tl['test_loss'], history_tl['test_acc'] = test[0].tolist(
        ), test[1].tolist()
        if not args.no_plot:
            plot_training(history_tl, args.model_dir)
        with open(os.path.join(args.model_dir, 'transfer_learn_history.txt'), 'w') as f:
            json.dump(history_tl, f)

    if args.fine_tune:
        model = None
        assert model.layers[FINE_TUNE_FINAL_LAYER_INDEX].name == FINE_TUNE_FINAL_LAYER_NAME
        set_trainable_layers(
            trainable_layer_list=model.layers[:
                                              FINE_TUNE_FINAL_LAYER_INDEX + 1],
            frozen_layer_list=model.layers[FINE_TUNE_FINAL_LAYER_INDEX + 1:])
        model.compile(optimizer=SGD(lr=0.00005, momentum=0.9),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history_ft = model.fit_generator(
            (x, y),
            epochs=nb_epoch,
            class_weight='auto')
        if not args.no_plot:
            plot_training(history_ft, args.model_dir)

    # model.save(os.path.join(args.model_dir, 'inceptionv3-ft.model'))


def plot_training(history, folder_dir):
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    test_acc = history['test_acc']
    test_loss = history['test_loss']
    epochs = range(len(acc))

    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': 'Arial',
                      'size': 10})
    width = 3.487
    height = width * 0.618
    fig, ax1 = plt.subplots()
    fig.set_size_inches(width, height)
    fig.subplots_adjust(left=.18, bottom=.2, right=.9, top=.95)

    ax1.plot(epochs, acc, 'g', label='training')
    ax1.plot(epochs, val_acc, 'b--', label='validation')
    ax1.plot(epochs[-1], test_acc, 'k*', label='testing')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    ax1.legend(loc='best')
    plt.savefig('transfer_learn_acc.png')
    filename = os.path.join(folder_dir, 'transfer_learn_acc')
    plt.savefig(filename + '.pdf')
    plt.savefig(filename + '.svg')
    plt.close(fig)
    subprocess.call(["explorer.exe", filename + ".pdf"])

    fig, ax1 = plt.subplots()
    fig.set_size_inches(width, height)
    fig.subplots_adjust(left=.18, bottom=.2, right=.9, top=.95)
    ax1.plot(epochs[1:], loss[1:], 'g', label='training')
    ax1.plot(epochs[1:], val_loss[1:], 'b--', label='validation')
    ax1.plot(epochs[-1], test_loss, 'k*', label='testing')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    ax1.legend(loc='best')
    plt.savefig('transfer_learn_loss.png')
    filename = os.path.join(folder_dir, 'transfer_learn_loss')
    plt.savefig(filename + '.pdf')
    plt.savefig(filename + '.svg')
    plt.close(fig)
    subprocess.call(["explorer.exe", filename + ".pdf"])


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument(
        "--image_dir", default=r'D:\NN\clothes_styles\warm_up_train_20180201\Images\skirt_length_labels')
    a.add_argument(
        "--model_dir", default=r'C:\tmp\InceptionResNet\skirt_length_labels')
    # a.add_argument(
    #     "--model_dir", default=r'C:\tmp\skirt_length_labels')
    a.add_argument("--image_lists", default='distorted_image_lists.json')
    # a.add_argument("--image_lists", default='image_lists.json')
    a.add_argument("--retrain_weights", default="retrain_weights.hdf5")
    a.add_argument("--nb_epoch", default=10000, type=int)
    a.add_argument("--batch_size", default=200, type=int)
    a.add_argument("--learning_rate", default=0.00005, type=int)
    a.add_argument("--earlystopping_patience", default=100, type=int)
    # a.add_argument("--val_batch_size", default=200)
    a.add_argument("--no_plot", default=False, action='store_true')
    a.add_argument("--transfer_learning", default=True)
    a.add_argument("--fine_tune", default=False)

    args = a.parse_args()
    # if (not os.path.exists(args.image_dir)):
    #     print("directories do not exist")
    #     sys.exit(1)

    main(args)
