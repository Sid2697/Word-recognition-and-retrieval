"""
This file contains code for performing word retrieval experiments
"""

# Standard Library imports
import pdb
import time
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm

# Third party imports
import Levenshtein as lev
from sklearn.neighbors import KDTree
from sklearn.metrics import average_precision_score

parser = argparse.ArgumentParser(description='Image features testing')

# Embeddings and text files paths
parser.add_argument('--text_features', default='embeddings/featsSynth.npy', help='numpy file containing text features')
parser.add_argument('--image_features', default='embeddings/featsImg.npy', help='numpy file containing image features')
parser.add_argument('--annotations_path', default='gen_files/ann_demo.txt', help='text file contaning annotations')
parser.add_argument('--ocr_opt_path', default='gen_files/ocr_output_demo.txt', help='text file contaning ocr output')
parser.add_argument('--master_dict', default='embeddings/master_dict_last_101_demo.pkl', help='Path to master dict pickle file')

# Different experiment's flag
parser.add_argument('--experiment_label', default='base', help='label to identify which experiment is going on [ocr_rank (Edit Distance on Text Recogniser Outputs), query_expand, naive_merge]')
parser.add_argument('--qbi', default=0, type=int, help='Count for doing number of time to perform Query by image to cover the entire data')
parser.add_argument('--query_by_image', default=False, help='If True query by image experiment is run, else query by text')

# Different results visulisation flags
parser.add_argument('--visual_text', default=False, help='If True saves a file with necessary information for visualization')
parser.add_argument('--visual_text_name', default=None, help='Name of file for visualization text file')
parser.add_argument('--get_stats', default=False, help='Flag for generating stats file')
parser.add_argument('--stats_name', default=None, help='Name of the stats file')

args = parser.parse_args()
print(args)

# Initialising logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s', filename='evaluation.log')
logging.info("Starting evaluation...")
logging.info(args)

def get_features(path_to_file):
    """
    This method is used to load the numpy arrays from saved files
    """
    return np.load(path_to_file, mmap_mode='r')


def get_annotations(path_to_file):
    """
    This method is used to load the annotations from a given text file
    """
    data = list()
    with open(path_to_file, 'r') as file:
        for line in file:
            data.append(line.split(' '))
    return data


def get_unique_words_index_list(path_annotation):
    """
    This method is used to get the list of unique words
    """
    pickle_file = args.master_dict
    annotations = get_annotations(path_annotation)
    unique_word_index = list()
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
    for key in tqdm(data.keys()):
        for annotation in annotations:
            if key in annotation:
                unique_word_index.append(annotation[-2])
                break
    return unique_word_index


def get_occurance_list(original_query, label_array):
    """
    This method gives a numpy array of number of times original query equals
    to the output
    original_query: query which is to be compared
    label_array: label 
    """
    output_list = np.zeros(len(label_array),)
    for i, label in enumerate(label_array):
        if label == original_query:
            output_list[i] = 1
    return output_list


path_image_features = args.image_features
path_text_features = args.text_features

print('[INFO] Loading embeddings...')
path_annotations = args.annotations_path
ocr_output_path = args.ocr_opt_path
image_features = get_features(path_image_features)
text_features = get_features(path_text_features)
text_features = np.nan_to_num(text_features)    # Removing Nan from invalid unigrams

print("[INFO] Creating KDtree...")
start = time.time()
kdt = KDTree(image_features, leaf_size=30, metric='euclidean')
print("[INFO] Time taken in creating KDtree {} seconds!".format(round(time.time() - start, 4)))


if args.query_by_image:
    print('[INFO] Generating queries...')
    all_queries_file = args.annotations_path
    with open(all_queries_file, 'r') as file:
        temp_data = file.readlines()
    queries_list = [item.split()[-2] for item in temp_data]
else:
    print("[INFO] Generating query list...")
    queries_list = get_unique_words_index_list(path_annotations)

queries = np.zeros((len(queries_list), text_features.shape[1]))

if not args.query_by_image:
    for i, query in enumerate(queries_list):
        queries[i, :] = text_features[int(query) - 1, :]
else:
    print('[INFO] Performing Query by Image for {} time...'.format(args.qbi + 1))
    initial = round(image_features.shape[0]/5) * args.qbi
    final = round(image_features.shape[0]/5) * (args.qbi + 1)
    queries = image_features[initial : final]
    queries_list = queries_list[initial : final]   

print('[INFO] Querying KDtree...')
start = time.time()
dist, ind = kdt.query(queries, k=image_features.shape[0], dualtree=True)
print('[INFO] Time taken in querying KDtree {} seconds!'.format(time.time() - start))

running_ap = 0.0
running_ap_inverse = 0.0
count = 0
correct_count = 0
total_count = 0

# Getting the annotations
annotations = get_annotations(path_annotations)
label_dict = dict()
for annotation in annotations:
    annotation_number = int(annotation[-2])
    if annotation_number not in label_dict.keys():
        label_dict[annotation_number] = annotation[1]

if args.visual_text or args.experiment_label=='ocr_rank' or args.experiment_label=='query_expand' or args.experiment_label=='naive_merge' or args.get_stats:
    annotations = get_annotations(ocr_output_path)
    ocr_dict = dict()
    for annotation in annotations:
        annotation_number = int(annotation[-2])
        if annotation_number not in ocr_dict.keys():
            ocr_dict[annotation_number] = annotation[-3]

if args.visual_text or args.get_stats:
    annotations = get_annotations(path_annotations)
    path_dict = dict()
    for annotation in annotations:
        annotation_number = int(annotation[-2])
        if annotation_number not in path_dict.keys():
            path_dict[annotation_number] = annotation[0]

if args.experiment_label == 'query_expand':
    start = time.time()
    first_indexs_list = list()
    first_indexs_dict = dict()
    for number, indexs in enumerate(ind):
        original_query_first = label_dict[int(queries_list[number])]
        present_in_ocr = False
        for number_, indexs_ in enumerate(indexs):
            if ocr_dict[indexs_ + 1] == original_query_first:
                first_indexs_list.append(indexs_ + 1)
                present_in_ocr = True
                first_indexs_dict[number] = (indexs_ + 1, present_in_ocr, number)
                break
        if not present_in_ocr:
            first_indexs_dict[number] = (indexs_ + 1, present_in_ocr, int(queries_list[number]))
            first_indexs_list.append(indexs[0] + 1)
    updated_queries = np.zeros((len(first_indexs_list), text_features.shape[1]))
    for count_ in sorted(first_indexs_dict.keys()):
        value = first_indexs_dict[count_]
        if value[1] == True:
            updated_queries[count_, :] = image_features[value[0] - 1, :]
        else:
            updated_queries[count_, :] = text_features[value[2] - 1, :]
    del dist
    del ind # Using lots of RAM.
    print('[INFO] Creating updated dist and ind...')
    updated_dist = np.zeros((updated_queries.shape[0], image_features.shape[0]))
    updated_ind = np.zeros((updated_queries.shape[0], image_features.shape[0]))
    for query_count, selected_query in enumerate(tqdm(updated_queries)):
        updated_dist[query_count], updated_ind[query_count] = kdt.query(selected_query.reshape(1, -1), k=image_features.shape[0], dualtree=True)
    dist = updated_dist
    ind = updated_ind
    print('[INFO] Updated dist and ind created...')
    print("[INFO] Time taken is ", time.time() - start)

start = time.time()

for i, indexs in enumerate(tqdm(ind, desc='Testing')):
    original_query = label_dict[int(queries_list[i])]

    label_array = list()
    if args.experiment_label == 'ocr_rank' or args.experiment_label == 'naive_merge':
        ocr_edit_distance_dict = dict()

    for index in indexs:
        index += 1
        label = label_dict[index]
        label_array.append(label)

        if args.experiment_label == 'ocr_rank' or args.experiment_label=='naive_merge':
            ocr_output = ocr_dict[index]
            lev_dist = lev.distance(original_query, ocr_output)
            if lev_dist not in ocr_edit_distance_dict.keys():
                ocr_edit_distance_dict[lev_dist] = [(ocr_output, index, label_dict[index], original_query)]
            else:
                ocr_edit_distance_dict[lev_dist].append((ocr_output, index, label_dict[index], original_query))

    if args.experiment_label == 'naive_merge':
        updated_label_array = list()
        updated_dist = list()
        used_indexs = list()
        zero_ed_list = list()
        if 0 in ocr_edit_distance_dict.keys():
            zero_ed_list = ocr_edit_distance_dict[0]
        for elements in zero_ed_list:
            updated_label_array.append(elements[2])
            position = np.where(indexs == elements[1] - 1)[0][0]
            updated_dist.append(0)
            used_indexs.append(elements[1])
        for index in indexs:
            index += 1
            if index in used_indexs:
                pass
            else:
                position = np.where(indexs == index - 1)[0][0]
                updated_dist.append(dist[i][position])
                updated_label_array.append(label_dict[index])
        dist[i] = updated_dist
        label_array = updated_label_array

    if args.experiment_label == 'ocr_rank':
        min_edit = min(ocr_edit_distance_dict.keys())
        max_edit = max(ocr_edit_distance_dict.keys())
        y_true_rank = list()
        y_label_rank = list()
        
        for edit_dist in sorted(ocr_edit_distance_dict.keys()):
            for entry in ocr_edit_distance_dict[edit_dist]:
                total_count += 1
                y_label_rank.append(max_edit - edit_dist)
                if entry[-2] == entry[0]:
                    correct_count += 1
                if entry[-2] == entry[-1]:
                    y_true_rank.append(1)
                else:
                    y_true_rank.append(0)
        y_label_rank_final = [x / max_edit for x in y_label_rank]   # normalizing the y_label_rank
        running_ap_inverse += average_precision_score(y_true_rank, y_label_rank_final)

    y_label = max(dist[i]) - dist[i]
    y_true = get_occurance_list(original_query, label_array)

    score = average_precision_score(y_true, y_label)
    
    if args.visual_text:
        file_name = 'output/' + args.visual_text_name + '.txt'
        with open(file_name, 'a') as visual_file:
            visual_file.write('Query: {} {} \n'.format(path_dict[int(queries_list[i])].split('/')[-1], original_query))
            visual_file.write('mAP: {} \n'.format(score))
            rank_count = 0
            for dist_count, index_temp in enumerate(indexs):
                index_temp += 1
                truth = y_true[dist_count]
                rank_count += 1
                visual_file.write('Rank: {} {} {} {} {} {} \n'.format(path_dict[index_temp].split('/')[-1], label_dict[index_temp], ocr_dict[index_temp], dist[i][dist_count], rank_count, truth))
                if rank_count == 100:
                    break
    if args.get_stats:
        file_name = 'output/' + args.stats_name + '.txt'
        with open(file_name, 'a') as stats_file:
            to_write = "Query {} mAP {} \n".format(original_query, round(score, 4))
            stats_file.write(to_write)
            rank_count = 0
            for dist_count, index_temp in enumerate(indexs):
                index_temp += 1
                truth = y_true[dist_count]
                rank_count += 1
                stats_file.write('Rank: {} {} {} {} {} {} \n'.format(path_dict[index_temp].split('/')[-1], label_dict[index_temp], ocr_dict[index_temp], dist[i][dist_count], rank_count, truth))
                if rank_count == 100:
                    break

    running_ap += score
    count += 1

    mAP = running_ap/count

if args.experiment_label == 'ocr_rank':
    print('[INFO] Mean average precision for experiment label {} is {}. And for the original experiment is {}'.format(args.experiment_label, running_ap_inverse/count, mAP))
    logging.info('[INFO] Mean average precision for experiment label {} is {}. And for the original experiment is {}'.format(args.experiment_label, running_ap_inverse/count, mAP))
else:
    print("[INFO] Mean average precision was: ", mAP)
    logging.info("[INFO] Mean average precision was: {}".format(mAP))

print('[INFO] Time taken in experiment was ', time.time() - start)
