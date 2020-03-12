"""
This file contains code for merging two embedding features
"""
import torch
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--image_embed_path', default='embeddings/featsImg.npy', help='Path to the image embeddings')
parser.add_argument('--ocr_embed_path', default='embeddings/featsSynth.npy', help='Path to the ocr embeddings')
parser.add_argument('--image_embed_weight', default=0.6, type=float, help='Weight to multiply by word image embedding')
parser.add_argument('--ocr_embed_weight', default=0.4, type=float, help='Weight to multiply by ocr prediction embedding')
parser.add_argument('--use_numpy_weight', default=False, help="Using numpy's inbuilt weighted average")
parser.add_argument('--get_max', default=False, help='Fuse matrices using max value')
parser.add_argument('--use_sum', default=False, help='Merging embedding using sum')

args = parser.parse_args()
print(args)

ocr_text_embedding_path = args.ocr_embed_path
word_image_embedding_path = args.image_embed_path
fused_embedding_path = 'output/'

print('[INFO] Loading Embeddings...')
ocr_text_embedding = np.load(ocr_text_embedding_path)
word_image_embedding = np.load(word_image_embedding_path)

def l2Normalize(inputTensor):
    normVal = torch.norm(inputTensor, p=2, dim=1).unsqueeze(1)
    if torch.isnan(normVal).sum().item()>0:
        print('Warning: Tensor having Nan:0')
        inputTensor[torch.isnan(inputTensor)] = 0.0 #NaN coming rarely when synthetic redering is failing for some unicode characters.
        normVal = torch.norm(inputTensor, p=2, dim=1).unsqueeze(1)
    normTensor = inputTensor.div(normVal.expand_as(inputTensor))
    return normTensor

new_embedding = np.zeros(ocr_text_embedding.shape)

if not args.use_sum:
    if not args.get_max:
        print('[INFO] Fusing embeddings using averaging...')
    else:
        print("[INFO] Fusing embeddings using max...")
    for embed_number in tqdm(range(ocr_text_embedding.shape[0])):
        if not args.use_numpy_weight:
            ocr_embed = ocr_text_embedding[embed_number] * args.ocr_embed_weight
            word_embed = word_image_embedding[embed_number] * args.image_embed_weight
            if args.get_max:
                average_embed = np.amax(np.array([ocr_embed, word_embed]), axis=0)
            else:
                average_embed = np.average(np.array([ocr_embed, word_embed]), axis=0)
        else:
            ocr_embed = ocr_text_embedding[embed_number]
            word_embed = word_image_embedding[embed_number]
            word_embed_weight = np.ones(word_embed.shape) * args.image_embed_weight
            ocr_embed_weight = np.ones(ocr_embed.shape) * args.ocr_embed_weight
            if args.get_max:
                average_embed = np.amax(np.array([ocr_embed, word_embed]), axis=0) 
            else:            
                average_embed = np.average(np.array([ocr_embed, word_embed]), weights=np.array([ocr_embed_weight, word_embed_weight]), axis=0)
        
        new_embedding[embed_number] = average_embed
else:
    print('[INFO] Using sum...')
    new_embedding = np.sum((word_image_embedding, ocr_text_embedding), axis=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('[INFO] Device name ', device)
new_embedding = torch.from_numpy(new_embedding).float().to(device)
new_embedding = l2Normalize(new_embedding)
new_embedding = new_embedding.cpu().numpy()

print('[INFO] Saving fused embeddings...')
if args.use_sum:
    np.save(fused_embedding_path + 'summed_fused_embedding.npy', new_embedding)
elif args.get_max:
    np.save(fused_embedding_path + 'max_fused_embedding_custom_weighted.npy', new_embedding)
else:
    np.save(fused_embedding_path + 'avg_fused_embedding_custom_weighted.npy', new_embedding)
