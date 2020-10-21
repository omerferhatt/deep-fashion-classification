import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical


def preprocess_list_category_img(path):
    list_category_img_f = open(path, 'r')

    total_row = list_category_img_f.readline().split()
    columns = list_category_img_f.readline().split()

    image_names = []
    categories = []
    category_headers = []
    for index, i in enumerate(list_category_img_f):
        image_names.append(i.split()[0])
        categories.append(i.split()[1])

    image_names = np.array(image_names, dtype=np.str)

    categories = np.array(categories, dtype=np.int)
    for i in set(categories):
        category_headers.append('category_{}'.format(i))

    columns = [columns[0]] + category_headers
    categories = to_categorical(categories)
    categories = np.delete(categories, obj=[0, 38, 45], axis=1)

    list_category_img_f.close()

    return image_names.reshape(-1, 1), categories, columns


def preprocess_list_bbox(path):
    list_bbox_f = open(path, 'r')
    total_row = list_bbox_f.readline().split()
    columns = list_bbox_f.readline().split()

    image_names = []
    bboxes = []

    for index, i in enumerate(list_bbox_f):
        image_names.append(i.split()[0])
        bboxes.append(i.split()[1:5])

    image_names = np.array(image_names, dtype=np.str)
    bboxes = np.array(bboxes, dtype=np.int16)

    list_bbox_f.close()

    return image_names.reshape(-1, 1), bboxes, columns


def preprocess_list_eval_partition(path):
    list_eval_partition = open(path, 'r')
    total_row = list_eval_partition.readline().split()
    columns = list_eval_partition.readline().split()

    image_names = []
    eval_status = []

    for index, i in enumerate(list_eval_partition):
        image_names.append(i.split()[0])
        eval_status.append(i.split()[1])

    image_names = np.array(image_names, dtype=np.str)
    eval_status = np.array(eval_status, dtype=np.str)

    list_eval_partition.close()

    return image_names.reshape(-1, 1), eval_status.reshape(-1, 1), columns


if __name__ == '__main__':
    image_names, bboxes, columns = preprocess_list_bbox('raw_txt/list_bbox.txt')
    _, categories, columns2 = preprocess_list_category_img('raw_txt/list_category_img.txt')
    _, eval_status, columns3 = preprocess_list_eval_partition('raw_txt/list_eval_partition.txt')
    df = pd.DataFrame(np.concatenate([image_names, eval_status, bboxes, categories], axis=1), columns=columns3+columns2[1:]+columns[1:])
    for t in ['train', 'val', 'test']:
        df[df['evaluation_status'] == t].to_csv('dataset_csv/list_combined_category_small_' + t + '.tsv', sep='\t',
                                                      index=False)
