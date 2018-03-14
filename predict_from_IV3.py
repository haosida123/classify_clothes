from keras.applications.inception_v3 import InceptionV3, preprocess_input
import os.path
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import requests
from io import BytesIO

IM_WIDTH, IM_HEIGHT = 299, 299  # fixed size for InceptionV3


def predict(model, img):
    """Run model prediction on image
    Args:
      model: keras model
      img: PIL format image
      target_size: (w,h) tuple
    Returns:
      list of predicted labels and their probabilities
    """
    target_size = (IM_WIDTH, IM_HEIGHT)
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(model.predict(x))
    preds = model.predict(x)[0]
    return preds


def plot_preds(image, preds):
    """Displays image and the top-n predicted probabilities in a bar graph
    Args:
      image: PIL image
      preds: list of predicted labels and their probabilities
    """
    plt.imshow(image)
    plt.axis('off')

    plt.figure()
    labels = ("cat", "dog")
    plt.barh(range(len(preds)), preds, alpha=0.5)
    plt.yticks([0, 1], labels)
    plt.xlabel('Probability')
    plt.xlim(0, 1.01)
    plt.tight_layout()
    plt.show()


def main(args):
    if os.path.exists(args.model_dir):
        model = load_model(args.model_dir)
        print('loading model from {}'.format(args.model_dir))
    else:
        print('creating new IV3 model...')
        model = InceptionV3()
        model.save(args.model_dir)
    if args.image_dir is not None:
        img = Image.open(args.image_dir)
        preds = predict(model, img)
        print(preds)
        plot_preds(img, preds)

    if args.image_url is not None:
        response = requests.get(args.image_url)
        img = Image.open(BytesIO(response.content))
        preds = predict(model, img)
        plot_preds(img, preds)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument(
        "--image_dir", default=r'C:\NN\clothes_styles\warm_up_train_20180201\Images\skirt_length_labels\0f5bb6069b8ef780217a32d93b670777.jpg')
    a.add_argument(
        "--model_dir", default=r'C:\tmp\warm_up_skirt_length\IV3.model')
    a.add_argument("--image_url", help="url to image")
    args = a.parse_args()
    main(args)
