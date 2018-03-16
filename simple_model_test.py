
import json
import numpy as np

# from keras.models import Sequential, Model
# from keras.layers import Dense, Activation  #, GlobalAveragePooling2D
# from keras.layers import Input


def main():
    model_dir = r'C:\tmp\warm_up_skirt_length'
    with open(model_dir + '/image_lists.json', 'r') as f:
        image_lists = json.load(f)
    n_classes = len(image_lists.keys())
    
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
