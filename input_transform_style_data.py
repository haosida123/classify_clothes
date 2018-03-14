# move the pics into sub-category folders as
# indicated by csv_file

import csv
import sys
import shutil
import os.path
from os import walk
import argparse
# from tensorflow.python.platform import gfile
import re
import hashlib
# from tensorflow.python.util import compat

LABEL_CAT = 'skirt_length_labels'
CAT_NUM = 6
EXTENSIONS = ['jpg', 'jpeg', 'JPG', 'JPEG']
IMAGE_DIR = 'C:/NN/clothes_styles/warm_up_train_20180201'  # path to /Image
CSV_DIR = IMAGE_DIR + r'/Annotations'
CSV_FILE = CSV_DIR + r'/{}.csv'.format(LABEL_CAT)
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def move_back_to_root(args):
    args.image_dir = args.image_dir + '/Images/' + args.label_cat
    for (dirpath, dirnames, filenames) in walk(args.image_dir):
        if os.path.samefile(dirpath, args.image_dir):
            continue
        for filename in filenames:
            if filename.split('.')[-1] in EXTENSIONS:
                move_from = os.path.join(dirpath, filename)
                move_to = os.path.join(args.image_dir, filename)
                shutil.move(move_from, move_to)


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.

    Args:
      dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def move(args):
    with open(args.csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        skipped = {'m_label': 0, 'error_category_num:': 0,
                   'y_not_found': 0, 'no_such_file': 0}
        total = 0
        for [jpg_file, label_category, label] in csv_reader:
            # 对于标签含m的数据，考虑到这部分标签数量较少，
            # 因此暂不将含这种类型的标签加入数据
            if label.find('m') != -1:
                skipped['m_label'] += 1
                continue
            # 跳过有问题的标签
            if len(label) != args.cat_num:
                skipped['error_category_num'] += 1
                continue

            category_index = label.find('y')
            if category_index == -1:
                print('y-label not found in file {}'.format(jpg_file))
                skipped['y_not_found'] += 1
                continue
            folder_name = 'label_' + str(category_index)
            path_splitted = jpg_file.split('/')
            jpg_folder = '/'.join(path_splitted[:-1])
            jpg_base = path_splitted[-1]
            move_from = '/'.join((args.image_dir, jpg_file))
            if not os.path.exists(move_from):
                skipped['no_such_file'] += 1
                continue

            hash_name = re.sub(r'_nohash_.*$', '', move_from)
            hash_name_hashed = hashlib.sha1(
                str.encode(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < args.validate_percent:
                folder_name = 'val/' + folder_name
                ensure_dir_exists('/'.join((args.image_dir, jpg_folder, 'val')))
            elif percentage_hash < (args.test_percent + args.validate_percent):
                folder_name = 'test/' + folder_name
                ensure_dir_exists('/'.join((args.image_dir, jpg_folder, 'test')))
            else:
                if args.test_percent or args.validate_percent:
                    folder_name = 'train/' + folder_name
                    ensure_dir_exists('/'.join((args.image_dir, jpg_folder, 'train')))
            move_to_dir = '/'.join((args.image_dir, jpg_folder, folder_name))
            ensure_dir_exists(move_to_dir)
            shutil.move(move_from, '/'.join((move_to_dir, jpg_base)))
            total += 1

        print('{} files skipped, {} files moved\nskip detail:'.
              format(sum(skipped.values()), total))
        print(skipped)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--validate_percent", default=10)
    a.add_argument("--test_percent", default=10)
    a.add_argument("--image_dir", default=IMAGE_DIR)
    a.add_argument("--label_cat", default=LABEL_CAT)
    a.add_argument("--csv_file", default=CSV_FILE)
    a.add_argument("--cat_num", default=CAT_NUM)
    a.add_argument("--move_back", action="store_true")
    args = a.parse_args()
    print(args.move_back)

    if not os.path.exists(args.image_dir):
        print("directories do not exist")
        sys.exit(1)
    if args.move_back:
        print('moving back...')
        move_back_to_root(args)
    else:
        if(not os.path.exists(args.csv_file)):
            print("csv file does not exist")
            sys.exit(1)
        move(args)
