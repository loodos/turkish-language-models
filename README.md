[English](./README.md) | [Türkçe](./README_TR.md)

# Turkish Language Models

## Introduction

In this repository, we publish Turkish language models and related tools.

# Updates

**June 29, 2020** Initial version of this repo

## Language Models

Electra -> https://github.com/google-research/electra
Albert -> https://github.com/google-research/ALBERT
Bert -> https://github.com/google-research/bert



## Result

## Trained Models

| Model        | Model type | Case    | Vocabulary Size | Train Data Size \(GB\) | Train Step \(Million\) |
|--------------|------------|---------|-----------------|----------------------------|----------------------------|
| 0\_electra | base       | uncased | 32              | 200                        | 1                          |
| 1\_electra | small      | uncased | 32              | 200                        | 1                          |
| 2\_electra | base       | uncased | 32              | 40                         | 1                          |
| 3\_electra | small      | uncased | 32              | 40                         | 1                          |
| 4\_electra | base       | uncased | 64              | 200                        | 1                          |
| 5\_bert    | base       | uncased | 32              | 40                         | 1                          |
| 6\_albert  | base       | uncased | 32              | 40                         | 1                          |


## Pretraining Details

Dataset -> collected from online blogs, newspapers -> can't share it due to copyright
You can use filtered common crawl corpus here -> https://oscar-corpus.com/



## Finetune Results

we used Farm (https://github.com/deepset-ai/FARM) for NER and sentiment analysis
QA is done with huggingface https://github.com/huggingface/transformers/tree/master/examples/question-answering

### NER

Training dataset is WikiAnn

The WikiANN dataset (Pan et al. 2017)  https://www.aclweb.org/anthology/P17-1178.pdf

Parameters for Farm Training
```
max_seq_len = 128
BATCH_SIZE = 16
EMBEDS_DROPOUT_PROB = 0.1
LEARNING_RATE = 1e-5
N_EPOCHS = 3
N_GPU = 1
```

| Model                                 | Farm Ner avg score |
|---------------------------------------|---------------|
| 0\_electra                            | 0\.86         |
| 1\_electra                            | 0\.58         |
| 2\_electra                            | 0\.84         |
| 3\_electra                            | 0\.77         |
| 4\_electra                            | 0\.86         |
| 5\_bert                               | 0\.84         |
| 6\_albert                             | 0\.83         |
| 10\_bert\_base\_128k\_uncased\_dbmdz  | 0\.84         |
| 11\_bert\_base\_32k\_uncased\_dbmdz   | 0\.83         |
| 13\_electra\_base\_32k\_cased\_dbmdz  | 0\.87         |
| 14\_electra\_small\_32k\_cased\_dbmdz | 0\.69         |


### QA

Using TQuad dataset -> https://github.com/TQuad/turkish-nlp-qa-dataset
we have modified it a bit to match Squad v2 format and removed some of the answers with misleading "answer_start" parameter

| Model                                 | Exact Score | F1 Score |
|---------------------------------------|-------------|----------|
| 0\_electra                            | 57\.41      | 76\.30   |
| 1\_electra                            | 28\.57      | 47\.00   |
| 2\_electra                            | 55\.10      | 75\.27   |
| 3\_electra                            | 44\.08      | 64\.15   |
| 4\_electra                            | 58\.91      | 78\.24   |
| 5\_bert                               | 53\.87      | 74\.66   |
| 6\_albert                             | 45\.30      | 66\.91   |
| 10\_bert\_base\_128k\_uncased\_dbmdz  | 22\.72      | 45\.57   |
| 11\_bert\_base\_32k\_uncased\_dbmdz   | 20\.40      | 43\.86   |
| 13\_electra\_base\_32k\_cased\_dbmdz  | 59\.59      | 79\.08   |
| 14\_electra\_small\_32k\_cased\_dbmdz | 37\.28      | 56\.06   |


### Sentiment

| Model                                 | Sentiment avg score |
|---------------------------------------|---------------------|
| 0\_electra                            | 0\.9197             |
| 1\_electra                            | 0\.8403             |
| 2\_electra                            | 0\.9143             |
| 3\_electra                            | 0\.8938             |
| 4\_electra                            | 0\.9167             |
| 5\_bert                               | 0\.9207             |
| 6\_albert                             | 0\.8900             |
| 10\_bert\_base\_128k\_uncased\_dbmdz  | 0\.9075             |
| 11\_bert\_base\_32k\_uncased\_dbmdz   | 0\.9048             |
| 13\_electra\_base\_32k\_cased\_dbmdz  | 0\.9320             |
| 14\_electra\_small\_32k\_cased\_dbmdz | 0\.8942             |


## Citation

## Acknowledgments

TQuad Dataset https://github.com/TQuad/turkish-nlp-qa-dataset
dbmdz models from stefan https://github.com/stefan-it/turkish-bert
sentiment dataset https://github.com/merveyapnaz/Sentiment-Analysist
TensorFlow Research Cloud (TFRC) for compute