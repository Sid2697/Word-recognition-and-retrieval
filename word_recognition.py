"""
This file has the code for calculating word accuracy using word embedding information
"""

# Standard library imports
import pdb
import pickle
import argparse

# Third party imports
import tqdm
import numpy as np
import Levenshtein as lev
from sklearn.neighbors import KDTree

parser = argparse.ArgumentParser()

# Embeddings and text file paths
parser.add_argument('--image_embeds', default='embeddings/topk_preds_100featsImg.npy', help='path to the image embeddings')
parser.add_argument('--topk_embeds', default='embeddings/topk_preds_100featsSynth.npy', help='path to the topk text embeds')
parser.add_argument('--predictions_file', default='gen_files/top_preds_embeds_100_with_conf.txt', help='path to the top preds text file options: [top_preds_embeds_with_confidence_1500, top_preds_embeds_all_with_confidence, top_preds_embeds_all_with_confidence_telugu_deep]')
parser.add_argument('--image_file', default='gen_files/image_embed_top_k_100.txt', help='path to the text file used for producing image embeddings options: [image_embed_top_k_1500, image_embed_top_k_all, test_ann_1000_pages_Telugu_deep]')

# Different experiments' flags
parser.add_argument('--use_confidence', default=False, help='If True we will use confidence score for re-ranking')

parser.add_argument('--k', default=20, type=int, help='Value of K')
args = parser.parse_args()

with open(args.predictions_file) as file:
    fileData = file.readlines()

predictions = [item.split()[-3] for item in fileData]
if args.use_confidence:
    confidenceScores = [1 - float(item.split()[-2]) for item in fileData]

with open(args.image_file) as file:
    file_data = file.readlines()
query = [item.split()[-3] for item in file_data]

print("[INFO] Loading word image and predictions' embeddings...")
image_embeds = np.load(args.image_embeds, mmap_mode='r')    # Enabling mmap_mode uses very very less RAM for loading the array as it uses the array directly from the disk
topk_embeds = np.load(args.topk_embeds, mmap_mode='r')

accuracyList = list()   # List for holding the accuracies
for i in range(args.k):     # Looping over top k predictions
    topk_count = 0  # Keeping track of TopK number
    correct = 0 # Keeping track of correct words
    total = 0   # Keeping track of total words tested
    use_ocr = 0
    use_other = 0
    # Looping over for calculating K for all K = 1, 2, ... K
    for count in tqdm.tqdm(range(len(image_embeds)), desc='[INFO] K = {}'.format(i + 1)):
        total += 1
        first_img_embed = image_embeds[count]   # Getting the first embedding
        corrs_topk_embeds = topk_embeds[topk_count : topk_count + i + 1]    # Getting top k embeddings corresponding to the first embedding
        kdt = KDTree(corrs_topk_embeds, leaf_size=30, metric='euclidean')   # Creating the KDTree for querying
        dist, ind = kdt.query(first_img_embed.reshape(1, -1), k=corrs_topk_embeds.shape[0], dualtree=True)  # Getting the distance and index by querying first embed on corresponding text
        # If we want to use the confidence scores
        if args.use_confidence:
            conf = list() # List for keeping track of the confidence scores
            for confCount in range(len(dist[0])):
                conf.append(confidenceScores[topk_count + ind[0][confCount]])
            updatedDist = conf + dist[0] # Updated distace value after considering the confidence scores
            newInd = ind[0][np.where(min(updatedDist) == updatedDist)[0][0]] # Updated index value after considering the confidence scores
            pred = predictions[topk_count + newInd]    # Updated predictions after considering the confidence scores
        else:
            try:
                pred = predictions[topk_count + ind[0][0]]
            except:
                pdb.set_trace()
        gt = query[count] # Getting the ground truth
        # Checking if the predicion equals the ground truth
        if lev.distance(gt, pred) == 0:
            correct += 1
        # Updating the top k count
        topk_count += 20
    accuracyList.append(correct/total * 100)
accuracyList = [round(item, 3) for item in accuracyList]
print('[INFO] Top {} accuracies are: {}'.format(len(accuracyList), accuracyList))
print('[INFO] Number of words tested on {}'.format(total))
