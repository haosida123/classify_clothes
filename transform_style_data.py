"""
move the pics into sub-category folders as 
indicated by csv_file
"""
import csv
import shutil
import os.path

LABEL_CAT = 'skirt_length_labels'
IMAGE_DIR = 'C:/NN/clothes_styles/warm_up_train_20180201' # path to /Image
CSV_DIR = IMAGE_DIR + r'/Annotations'
CSV_FILE = CSV_DIR + r'/{}.csv'.format(LABEL_CAT)




def move_files_by_catagory(label_data_frame):
    for label in label_data_frame[2]:

if __name__ == "__main__":
    assert os.path.exists(CSV_FILE)
    data = pandas.read_csv(CSV_FILE, header=None)
    # TODO: create sub-folders necessary?
    with open(CSV_FILE, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for [jpg_file, label_category, label] in csv_reader:
            print(row)

            # 对于标签含m的数据，考虑到这部分标签数量较少，
            # 因此暂不将含这种类型的标签加入数据
            if label.find('m') != -1:
                continue

            category_index = label.find('y')
            assert category_index != -1
            folder_name = 'label_' + str(category_index)

