Fused Text Recogniser and Deep Embeddings Improve Word Recognition and Retrieval
=================================================================================
<!-- TODO: Add the link to archive paper -->
<!-- [![arXiv](https://img.shields.io/badge/cs.cv-arXiv%3A1908.05849-42ba94.svg)](http://arxiv.org/abs/1908.05849) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

### [Project page](https://sid2697.github.io/Word-recognition-and-retrieval/) | [Demonstration](https://sid2697.github.io/files/Word_Retrieval_demo.gif) | [Poster](https://sid2697.github.io/files/Siddhant_Bansal_V4.pdf) | Paper

This repository contains code for the paper

"**Fused Text Recogniser and Deep Embeddings Improve Word Recognition and Retrieval**" *[Siddhant Bansal](https://sid2697.github.io), [Praveen Krishnan](https://kris314.github.io), [C.V. Jawahar](https://faculty.iiit.ac.in/~jawahar/index.html)* 
published in DAS 2020.

## Word Recognition Results
![Word Recognition](resources/WordRecVisualisation.png)

## Word Retrieval Demonstration
<!-- ![Word Retrieval](resources/Word_Retrieval_demo.gif) -->
[![IMAGE ALT TEXT](resources/Thumbnail.png)](https://youtu.be/3_InpQoJCMg)

Usage
-----------
### Cloning the repository
```
git clone https://github.com/Sid2697/Word-recognition-and-retrieval.git
```
### Install Pre-requisites
- Python >= 3.5
- PyTorch
- Scikit-learn
- NumPy
- tqdm

**`requirements.txt`** has been provided for installing Python dependencies.

```sh
pip install -r requirements.txt
```
### Generating/using deep embeddings
The deep embeddings used in this work are generated using the End2End network proposed in:
```
Krishnan, P., Dutta, K., Jawahar, C.V.: Word spotting and recognition using deep embedding. In: 2018 13th IAPR International Workshop on Document Analysis Systems (DAS). pp. 1–6 (April 2018). https://doi.org/10.1109/DAS.2018.70
```
Word text and image's deep embeddings for testing this repository are provided in the ```embeddings``` folder.
Text files containing the information about the embeddings are required while running the code. They are in the format<br>
```<img1-path><space><text1-string><space><dummyInt><space>1```<br>
```<img2-path><space><text2-string><space><dummyInt><space>1```<br>
...<br>
Corresponding text files for testing this repository are provided in the ``gen_files`` folder.

### Running the code
For running word recognition use the command:
```sh
python word_recognition.py
```
For running word recognition with confidence score use the command:
```sh
python word_recognition.py --use_confidence=True
```
Other arguments for word recognition experiment are:
```sh
--image_embeds
--topk_embeds
--predictions_file
--image_file
```
- `image_embeds` is used to provide path to the image embeddings
- `topk_embeds` is used to provide path to the TopK predictions' embeddings
- `image_file` is used to provide path to the image's text information file
- `predictions_file` is used to provide path to the TopK predictions' text information file

For running word retrieval use the command:
```sh
python word_retrieval.py
```
For running word retrieval's naive merge experiment use the command:
```sh
python word_retrieval.py --experiment_label=naive_merge
```
Other options for `experiment_label` are: `ocr_rank` and `query_expand`

Other major arguments for word retrieval experiment are:
- `text_features` is used to provide path to the text embeddings
- `image_features` is used to provide path to the image embeddings
- `annotations_path` is used to provide path to the text file containing annotations
- `ocr_opt_path` is used to provide path to the text file containing OCR predictions

TODO
-----------
- Add citation information
- Add link to arXiv paper

Contact
-----------
In case of any query contact [Siddhant Bansal](https://sid2697.github.io).
