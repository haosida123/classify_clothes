"""
move the pics into sub-category folders as
indicated by csv_file
"""
import csv
import shutil
import os.path

LABEL_CAT = 'skirt_length_labels'
CAT_NUM = 6
IMAGE_DIR = 'C:/NN/clothes_styles/warm_up_train_20180201'  # path to /Image
CSV_DIR = IMAGE_DIR + r'/Annotations'
CSV_FILE = CSV_DIR + r'/{}.csv'.format(LABEL_CAT)


def main():
    assert os.path.exists(CSV_FILE)
    with open(CSV_FILE, 'r') as csvfile:
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
            if len(label) != CAT_NUM:
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
            move_from = '/'.join((IMAGE_DIR, jpg_file))
            move_to_dir = '/'.join((IMAGE_DIR, jpg_folder, folder_name))
            if not os.path.exists(move_from):
                skipped['no_such_file'] += 1
                continue
            if not os.path.exists(move_to_dir):
                os.mkdir(move_to_dir)
            shutil.move(move_from, '/'.join((move_to_dir, jpg_base)))
            total += 1

        print('{} files skipped, {} files moved\nskip detail:'.
              format(sum(skipped.values()), total))
        print(skipped)


if __name__ == "__main__":
    main()
