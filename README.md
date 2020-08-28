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

Electra -> https://github.com/google-research/electra

Albert -> https://github.com/google-research/ALBERT

Bert -> https://github.com/google-research/bert

# Trained Models

|         Model                     | Vocabulary Size (Thousand) | Train Raw Text Data Size (GB) |     Train Step (Million)   |
|-----------------------------------|-----------------|----------------------|-------------------------------------|
|[electra-small-turkish-cased-discriminator](https://github.com/Loodos/transformers/tree/master/model_cards/loodos/electra-small-turkish-cased-discriminator)       | 32              | 200                  | 1                                   | 
|[electra-small-turkish-uncased-discriminator](https://github.com/Loodos/transformers/tree/master/model_cards/loodos/electra-small-turkish-uncased-discriminator)      | 32              | 40                   | 1                                   | 
|[electra-base-turkish-uncased-discriminator](https://github.com/Loodos/transformers/tree/master/model_cards/loodos/electra-base-turkish-uncased)     | 32              | 40                   | 1                                   | 
|[electra-base-turkish-64k-uncased-discriminator](https://github.com/Loodos/transformers/tree/master/model_cards/loodos/electra-base-turkish-64k-uncased-discriminator)   | 64              | 200                  | 1                                   | 
|[bert-base-turkish-uncased](https://github.com/Loodos/transformers/tree/master/model_cards/loodos/bert-base-turkish-uncased)          | 32              | 40                   | 3                                   |
|[albert-base-turkish-uncased](https://github.com/Loodos/transformers/tree/master/model_cards/loodos/albert-base-turkish-uncased)       | 32              | 40                   | 3                                   |


## Pretraining Details

* Our training dataset(totally 200 GB raw Turkish text) is collected from online blogs, free e-books, newspapers, common crawl corpuses, Twitter, articles, Wikipedia and so.

* We can not able to share all dataset due to copyright.
But we are planning to share some part of it. Check our [website](https://www.loodos.com.tr/) frequently.

* You can use filtered common crawl OSCAR corpus from [here](https://oscar-corpus.com/).

* On pretraining and finetuning, we have found a normalization issue specific to Turkish in both Google's and Huggingface's repos.
You can check details from [here](https://github.com/huggingface/transformers/issues/6680).
We are waiting Huggingface for solving this issue. Until it is solved, we significantly suggest you to use our [TextNormalization module](https://github.com/Loodos/turkish-language-models/blob/master/text_normalization.py) before tokenizing.

## Finetune Results

We used [Farm](https://github.com/deepset-ai/FARM) for NER and sentiment analysis.
QA is done with [Huggingface QA](https://github.com/huggingface/transformers/tree/master/examples/question-answering).

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
| electra-small-turkish-cased-discriminator                          | 0.58               | -         |
| electra-small-turkish-uncased-discriminator                        | 0.77               | 0.57      |
| electra-base-turkish-uncased-discriminator                         | 0.84               | 0.74      |
| electra-base-turkish-64k-uncased-discriminator                     | 0.86               | 0.73      |
| bert-base-turkish-uncased                            | 0.84               | 0.90      |
| albert-base-turkish-uncased                          | 0.83               | 0.87      |
| dbmdz/bert-base-turkish-128k-uncased                 | 0.84               | 0.76      |
| dbmdz/bert-base-turkish-uncased                      | 0.83               | 0.74      |
| dbmdz/electra-base-turkish-cased-discriminator       | 0.87               | 0.74      |
| dbmdz/electra-small-turkish-cased-discriminator-discriminator      | 0.69               | 0.50      |



### QA

Training dataset is [TQuad](https://github.com/TQuad/turkish-nlp-qa-dataset).
 
* We have modified it a bit to match Squad v2 format and removed some of the answers with misleading "answer_start" parameter.

| Model                                                       | Exact Score | F1 Score |
|-------------------------------------------------------------|-------------|----------|
| electra-small-turkish-cased-discriminator                                 | 28.57       | 47.00    |
| electra-small-turkish-uncased-discriminator                               | 44.08       | 64.15    |
| electra-base-turkish-uncased-discriminator                                | 55.10       | 75.27    |
| electra-base-turkish-64k-uncased-discriminator                            | 58.91       | 78.24    |
| bert-base-turkish-uncased                                   | 53.87       | 74.66    |
| albert-base-turkish-uncased                                 | 45.30       | 66.91    |
| dbmdz/bert-base-turkish-128k-uncased                        | 22.72       | 45.57    |
| dbmdz/bert-base-turkish-uncased                             | 20.40       | 43.86    |
| dbmdz/electra-base-turkish-cased-discriminator              | 59.59       | 79.08    |
| dbmdz/electra-small-turkish-cased-discriminator-discriminator             | 37.28       | 56.06    |


### Sentiment

Training dataset is [Sentiment Analysist](https://github.com/merveyapnaz/Sentiment-Analysist).

|                     Model                                 | Sentiment avg score |
|-----------------------------------------------------------|---------------------|
| electra-small-turkish-cased-discriminator                               | 0.8403              |
| electra-small-turkish-uncased-discriminator                             | 0.8938              |
| electra-base-turkish-uncased-discriminator                              | 0.9143              |
| electra-base-turkish-64k-uncased-discriminator                          | 0.9167              |
| bert-base-turkish-uncased                                 | 0.9207              |
| albert-base-turkish-uncased                               | 0.8900              |
| dbmdz/bert-base-turkish-128k-uncased                      | 0.9075              |
| dbmdz/bert-base-turkish-uncased                           | 0.9048              |
| dbmdz/electra-base-turkish-cased-discriminator            | 0.9320              |
| dbmdz/electra-small-turkish-cased-discriminator-discriminator           | 0.8942              |


## Citation

We are preparing a paper about our models and results. Until it is published, you can give this repository in your citations.

## Special Thanks

We are thankful to [TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc) for providing TPUs for pretraining our models,
to [DBMDZ](https://github.com/stefan-it/turkish-bert) for their detailed pretraining cheatsheet and [Prof. Dr. Banu Diri](https://avesis.yildiz.edu.tr/diri) for guidance and providing some datasets. 
