import json
import os.path
import argparse
from keras_fine_tune import plot_training


def main(args):
    if args.transfer_learning:
        with open(os.path.join(
                args.model_dir, 'transfer_learn_history.txt'), 'r') as f:
            history_tl = json.load(f)
        plot_training(history_tl, args.model_dir)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument(
        "--image_dir", default=r'D:\NN\clothes_styles\base\Images\skirt_length_labels')
    a.add_argument(
        "--model_dir", default=r'C:\tmp\pant_length_labels')
    a.add_argument("--transfer_learning", default=True)
    a.add_argument("--fine_tune", default=False)

    args = a.parse_args()
    # if (not os.path.exists(args.image_dir)):
    #     print("directories do not exist")
    #     sys.exit(1)

    main(args)
