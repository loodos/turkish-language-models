# Turkish Language Models

## Introduction

In this repository, we publish Transformer based Turkish language models and related tools.

New models, datasets, updates and tutorials are on the way. Please keep in touch with [us](https://www.loodos.com.tr/).

For any question or request, feel free to open an issue.
We are an enthusiastic R&D team to move forward Turkish NLP community and we need your feedbacks.

You can also check our [Zemberek-Python](https://github.com/Loodos/zemberek-python) implementation. It is fully Python integrated and
neither Java nor JVM are required. 

# Updates

**August 28, 2020** Initial version of this repo.


## Language Models

[Electra](https://github.com/google-research/electra), [ALBERT](https://github.com/google-research/ALBERT), [BERT](https://github.com/google-research/bert)

# Trained Models

|         Model                     | Vocabulary Size (Thousand) | Train Raw Text Data Size (GB) |     Train Step (Million)   |
|-----------------------------------|-----------------|----------------------|-------------------------------------|
|[loodos/electra-small-turkish-cased-discriminator](https://huggingface.co/loodos/electra-small-turkish-cased-discriminator)       | 32              | 200                  | 1                                   | 
|[loodos/electra-small-turkish-uncased-discriminator](https://huggingface.co/loodos/electra-small-turkish-uncased-discriminator)      | 32              | 40                   | 1                                   | 
|[loodos/electra-base-turkish-uncased-discriminator](https://huggingface.co/loodos/electra-base-turkish-uncased-discriminator)     | 32              | 40                   | 1                                   | 
|[loodos/electra-base-turkish-64k-uncased-discriminator](https://huggingface.co/loodos/electra-base-turkish-64k-uncased-discriminator)   | 64              | 200                  | 1                                   | 
|[loodos/bert-base-turkish-uncased](https://huggingface.co/loodos/bert-base-turkish-uncased)          | 32              | 40                   | 3                                   |
|[loodos/albert-base-turkish-uncased](https://huggingface.co/loodos/albert-base-turkish-uncased)       | 32              | 40                   | 3                                   |


## Pretraining Details

* Our training dataset(totally 200 GB raw Turkish text) is collected from online blogs, free e-books, newspapers, common crawl corpuses, Twitter, articles, Wikipedia and so.

* We are not able to share all of our dataset due to copyright issues.
But we are planning to share some part of it. Check our [website](https://www.loodos.com.tr/) frequently.

* You can use filtered common crawl OSCAR corpus from [here](https://oscar-corpus.com/).

* On pretraining and finetuning, we have found a normalization issue specific to Turkish in both Google's and Huggingface's repos.
You can check details from [here](https://github.com/huggingface/transformers/issues/6680).
We are waiting Huggingface for solving this issue. Until it is solved, we significantly suggest you to use our [TextNormalization module](https://github.com/Loodos/turkish-language-models/blob/master/text_normalization.py) before tokenizing.

## Finetune Results

We used [Farm](https://github.com/deepset-ai/FARM) for NER and sentiment analysis.
QA is done with [Huggingface QA](https://github.com/huggingface/transformers/tree/master/examples/question-answering).
DBMDZ models are also known as BERTurk.

### NER

Training dataset is [The WikiANN dataset (Pan et al. 2017)](https://www.aclweb.org/anthology/P17-1178.pdf).

Parameters for Farm Training
```
max_seq_len = 128
BATCH_SIZE = 16
EMBEDS_DROPOUT_PROB = 0.1
LEARNING_RATE = 1e-5
N_EPOCHS = 3
N_GPU = 1
```

| Model                                                | Farm Ner avg score | Dataset 2 |
|------------------------------------------------------|--------------------|-----------|
| loodos/electra-small-turkish-cased-discriminator                          | 0.58               | -         |
| loodos/electra-small-turkish-uncased-discriminator                        | 0.77               | 0.57      |
| loodos/electra-base-turkish-uncased-discriminator                         | 0.84               | 0.74      |
| loodos/electra-base-turkish-64k-uncased-discriminator                     | 0.86               | 0.73      |
| loodos/bert-base-turkish-uncased                            | 0.84               | 0.90      |
| loodos/albert-base-turkish-uncased                          | 0.83               | 0.87      |
| dbmdz/bert-base-turkish-128k-uncased                 | 0.84               | 0.76      |
| dbmdz/bert-base-turkish-uncased                      | 0.83               | 0.74      |
| dbmdz/electra-base-turkish-cased-discriminator       | 0.87               | 0.74      |
| dbmdz/electra-small-turkish-cased-discriminator      | 0.69               | 0.50      |



### QA

Training dataset is [TQuad](https://github.com/TQuad/turkish-nlp-qa-dataset).
 
* We have modified it a bit to match Squad v2 format and removed some of the answers with misleading "answer_start" parameter.

| Model                                                       | Exact Score | F1 Score |
|-------------------------------------------------------------|-------------|----------|
| loodos/electra-small-turkish-cased-discriminator                                 | 28.57       | 47.00    |
| loodos/electra-small-turkish-uncased-discriminator                               | 44.08       | 64.15    |
| loodos/electra-base-turkish-uncased-discriminator                                | 55.10       | 75.27    |
| loodos/electra-base-turkish-64k-uncased-discriminator                            | 58.91       | 78.24    |
| loodos/bert-base-turkish-uncased                                   | 53.87       | 74.66    |
| loodos/albert-base-turkish-uncased                                 | 45.30       | 66.91    |
| dbmdz/bert-base-turkish-128k-uncased                        | 22.72       | 45.57    |
| dbmdz/bert-base-turkish-uncased                             | 20.40       | 43.86    |
| dbmdz/electra-base-turkish-cased-discriminator              | 59.59       | 79.08    |
| dbmdz/electra-small-turkish-cased-discriminator             | 37.28       | 56.06    |


### Sentiment

Training dataset is [Sentiment Analysist](https://github.com/merveyapnaz/Sentiment-Analysist).

|                     Model                                 | Sentiment avg score |
|-----------------------------------------------------------|---------------------|
| loodos/electra-small-turkish-cased-discriminator                               | 0.8403              |
| loodos/electra-small-turkish-uncased-discriminator                             | 0.8938              |
| loodos/electra-base-turkish-uncased-discriminator                              | 0.9143              |
| loodos/electra-base-turkish-64k-uncased-discriminator                          | 0.9167              |
| loodos/bert-base-turkish-uncased                                 | 0.9207              |
| loodos/albert-base-turkish-uncased                               | 0.8900              |
| dbmdz/bert-base-turkish-128k-uncased                      | 0.9075              |
| dbmdz/bert-base-turkish-uncased                           | 0.9048              |
| dbmdz/electra-base-turkish-cased-discriminator            | 0.9320              |
| dbmdz/electra-small-turkish-cased-discriminator           | 0.8942              |


## Citation

We are preparing a paper about our models and results. Until it is published, you can give this repository in your citations.

## Special Thanks

We are thankful to [TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc) for providing TPUs for pretraining our models,
to [DBMDZ](https://github.com/stefan-it/turkish-bert) for their detailed pretraining cheatsheet and [Prof. Dr. Banu Diri](https://avesis.yildiz.edu.tr/diri) for  her guidance and providing some datasets. 
