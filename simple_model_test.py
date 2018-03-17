
# import json
# import numpy as np
from keras_fine_tune import add_final_layer
# from keras.models import Sequential, Model
# from keras.layers import Dense, Activation  #, GlobalAveragePooling2D
from keras.layers import Input


def main():
    weights_file = 'retrain_weights.hdf5'
    retrain_input_tensor = Input(shape=(2048,))
    retrain_model = add_final_layer(
        retrain_input_tensor, retrain_input_tensor, 6)
    print('loading weights: {}'.format(weights_file))
    retrain_model.load_weights(weights_file)
    weights = retrain_model.get_weights()
    print(weights)
    retrain_model.layers[-1].set_weights(weights[-1])
    for layer in retrain_model.layers:
        print(layer.name)
        print(layer)
    # inp = Input(shape=(2,))
    # x = Dense(5, activation='relu')(inp)
    # x = Dense(2, activation='softmax')(x)
    # model = Model(inputs=inp, outputs=x)

    # model.compile(loss='sparse_categorical_crossentropy',
    #             optimizer='rmsprop',
    #             metrics=['accuracy'])

    # inp = np.array([[2, 3],[3, 2],[1, 3],[0, 3],[18, 19]])
    # out = np.array([[1],[0],[1],[1],[1]])

    # model.fit(inp, out)

    # print(model.predict(np.array([[8,9]])))

    # inp = Input(shape=(2,))
    # x = Dense(5, activation='relu')(inp)
    # x = Dense(2, activation='softmax')(x)
    # model = Model(inputs=inp, outputs=x)

    # model.compile(loss='categorical_crossentropy',
    #             optimizer='rmsprop',
    #             metrics=['accuracy'])

    # inp = np.array([[2, 3],[3, 2],[1, 3],[0, 3],[18, 19]])
    # out = np.array([[1, 0],[0, 1],[1, 0],[1, 0],[1, 0]])

    # model.fit(inp, out)

    # print(model.predict(np.array([[8,9]])))


main()
