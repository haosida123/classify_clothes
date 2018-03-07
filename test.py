import csv

LABEL_CAT = 'skirt_length_labels'
IMAGE_DIR = r'C:\NN\clothes_styles\warm_up_train_20180201'  # path to \Image
CSV_DIR = IMAGE_DIR + r'\Annotations'
CSV_FILE = CSV_DIR + r'\{}.csv'.format(LABEL_CAT)


with open(CSV_FILE, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        print(row)
