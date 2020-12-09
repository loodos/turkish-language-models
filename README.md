# Turkish Language Models

## Introduction

In this repository, we publish Transformer based Turkish language models and related tools.

New models, datasets, updates and tutorials are on the way. Please keep in touch with [us](https://www.loodos.com.tr/).

For any question or request, feel free to open an issue.
We are an enthusiastic R&D team to contribute to Turkish NLP community and we need your feedbacks.

You can also check our [Zemberek-Python](https://github.com/Loodos/zemberek-python) implementation. It is fully Python integrated and
neither Java nor JVM are required. 

# Updates

**August 28, 2020** Initial version of this repo.

**December 8, 2020** New models have been trained with different corpus and training parameters. Finetuning scores have been updated with optimal hyper parameters.
* Bert-cased is newly added.
* Bert-uncased is updated.

## Language Models

[ELECTRA](https://github.com/google-research/electra), [ALBERT](https://github.com/google-research/ALBERT), [BERT](https://github.com/google-research/bert)

# Trained Models

|         Model                     | Vocabulary Size (Thousand) | Train Raw Text Data Size (GB) |     Train Step (Million)   |
|-----------------------------------|-----------------|----------------------|-------------------------------------|
|[loodos/electra-small-turkish-cased-discriminator](https://huggingface.co/loodos/electra-small-turkish-cased-discriminator)       | 32              | 200                  | 1                                   | 
|[loodos/electra-small-turkish-uncased-discriminator](https://huggingface.co/loodos/electra-small-turkish-uncased-discriminator)      | 32              | 40                   | 1                                   | 
|[loodos/electra-base-turkish-uncased-discriminator](https://huggingface.co/loodos/electra-base-turkish-uncased-discriminator)     | 32              | 40                   | 1                                   | 
|[loodos/electra-base-turkish-64k-uncased-discriminator](https://huggingface.co/loodos/electra-base-turkish-64k-uncased-discriminator)   | 64              | 200                  | 1                                   | 
|[loodos/bert-base-turkish-uncased](https://huggingface.co/loodos/bert-base-turkish-uncased)          | 32              | 40                   | 5                                   |
|[loodos/bert-base-turkish-cased](https://huggingface.co/loodos/bert-base-turkish-cased)          | 32              | 40                   | 5                                   |
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

Results are recorded with our super computer, called Pantheon.

System Specs:
* AMD Ryzen 3950x 
* 2 x Titan RTX
* 128 GB Ram
* 4 TB SSD and 30 TB HDD
* MSI x570 Godlike Motherboard
* Ubuntu 18.04, Cuda 10.1, Python 3.8.6, Tensorflow 2.3.1, Pytorch 1.7.0, Transformers 3.4.0

DBMDZ models are also known as BERTurk.

### NER

Huggingface's token classification [example](https://github.com/huggingface/transformers/tree/master/examples/token-classification) is used for scoring.

Our training dataset is the same dataset that has been used by "Mustafa Keskin, Banu Diri, “Otomatik Veri Etiketleme ile Varlık ̇Ismi Tanıma”, 4st
International Mediterranean Science and Engineering Congress (IMSEC 2019),
322-326."

You can contact [Prof. Dr. Banu Diri](https://avesis.yildiz.edu.tr/diri) for reaching the dataset and other details of the conference paper.


Hyper parameters(others are default):
```
task_type: NER
max_seq_len: 512
learning_rate: 5e-5
num_train_epochs: 5
seed: 1
per_gpu_train_batch_size: depends on model
```


| Model                                                                     | F1 score |
|---------------------------------------------------------------------------|----------|
| loodos/electra-small-turkish-cased-discriminator                          | 0.63     |
| loodos/electra-small-turkish-uncased-discriminator                        | 0.81     |
| loodos/electra-base-turkish-uncased-discriminator                         | 0.87     |
| loodos/electra-base-turkish-64k-uncased-discriminator                     | 0.86     |
| loodos/bert-base-turkish-uncased                                          | 0.89     |
| loodos/bert-base-turkish-cased                                            | 0.90     |
| loodos/albert-base-turkish-uncased                                        | 0.85     |
| dbmdz/bert-base-turkish-128k-cased                                        | 0.89     |
| dbmdz/bert-base-turkish-128k-uncased                                      | 0.80     |
| dbmdz/bert-base-turkish-uncased                                           | 0.89     |
| dbmdz/bert-base-turkish-cased                                             | 0.90     |
| dbmdz/electra-base-turkish-cased-discriminator                            | 0.89     |
| dbmdz/electra-small-turkish-cased-discriminator                           | 0.79     |



### QA

Huggingface's question-answering [example](https://github.com/huggingface/transformers/tree/master/examples/question-answering) is used for scoring.

Training dataset is [TQuad](https://github.com/TQuad/turkish-nlp-qa-dataset).
 
* We have modified it a bit to match Squad v2 format and removed some of the answers with misleading "answer_start" parameter.

Hyper parameters(others are default):
```
max_seq_len: 512
doc_stride: 128
learning_rate: 3e-5
num_train_epochs: 5
per_gpu_train_batch_size: depends on model
```

| Model                                                       | Exact Score | F1 Score |
|-------------------------------------------------------------|-------------|----------|
| loodos/electra-small-turkish-cased-discriminator            | 28.57       | 47.00    |
| loodos/electra-small-turkish-uncased-discriminator          | 36.09       | 57.68    |
| loodos/electra-base-turkish-uncased-discriminator           | 54.82       | 73.75    |
| loodos/electra-base-turkish-64k-uncased-discriminator       | 54.70       | 75.39    |
| loodos/bert-base-turkish-uncased                            | 58.40       | 75.86    |
| loodos/bert-base-turkish-cased                              | 58.29       | 76.37    |
| loodos/albert-base-turkish-uncased                          | 46.63       | 66.31    |
| dbmdz/bert-base-turkish-128k-uncased                        | 59.41       | 77.50    |
| dbmdz/bert-base-turkish-128k-cased                          | 60.53       | 77.49    |
| dbmdz/bert-base-turkish-uncased                             | 59.75       | 76.48    |
| dbmdz/bert-base-turkish-cased                               | 58.40       | 76.19    |
| dbmdz/electra-base-turkish-cased-discriminator              | 57.39       | 77.51    |
| dbmdz/electra-small-turkish-cased-discriminator             | 31.61       | 53.08    |


### Sentiment Analysis

Our custom text classification project is used for scoring.
TFAlbertForSequenceClassification, TFBertForSequenceClassification and TFElectraForSequenceClassification of Huggingface's Transformers are used in that project.

Training dataset is [Sentiment Analysist](https://github.com/merveyapnaz/Sentiment-Analysist).
We have normalized each sentence in the dataset with Zemberek for better scoring before training. The samples were split to %80 train, %20 test.

Hyper parameters(others are default):
```
max_seq_len: 192
learning_rate: 1e-6
epsilon: 1e-7
num_train_epochs: 15
per_gpu_train_batch_size: depends on model
```

|                     Model                                 | Test score |
|-----------------------------------------------------------|------------|
| loodos/electra-small-turkish-cased-discriminator          | 66.22     |
| loodos/electra-small-turkish-uncased-discriminator        | 78.97     |
| loodos/electra-base-turkish-uncased-discriminator         | 89.70     |
| loodos/electra-base-turkish-64k-uncased-discriminator     | 88.17     |
| loodos/bert-base-turkish-uncased                          | 92.17     |
| loodos/bert-base-turkish-cased                            | 91.52     |
| loodos/albert-base-turkish-uncased                        | 88.59     |
| dbmdz/bert-base-turkish-128k-uncased                      | 91.92     |
| dbmdz/bert-base-turkish-128k-cased                        | 91.94     |
| dbmdz/bert-base-turkish-uncased                           | 91.02     |
| dbmdz/bert-base-turkish-cased                             | 91.67     |
| dbmdz/electra-base-turkish-cased-discriminator            | 91.85     |
| dbmdz/electra-small-turkish-cased-discriminator           | 83.70     |


## Citation

We are preparing a paper about our models and results. Until it is published, you can give this repository in your citations.

## Special Thanks

We are thankful to [TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc) for providing TPUs for pretraining our models,
to [DBMDZ](https://github.com/stefan-it/turkish-bert) for their detailed pretraining cheatsheet and [Prof. Dr. Banu Diri](https://avesis.yildiz.edu.tr/diri) for  her guidance and providing some datasets. 
