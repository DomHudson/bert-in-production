# Bert in Production

A small collection of resources on using BERT (https://arxiv.org/abs/1810.04805
) and related Language Models in production environments.

## Descriptive Resources - how BERT works.

- [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

- [Sequence-to-sequence modeling with `NN.TRANSFORMER` and `TORCHTEXT`](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

- [Exploring BERT's Vocabulary](http://juditacs.github.io/2019/02/19/bert-tokenization-stats.html)

- [BERT to the rescue!](https://towardsdatascience.com/bert-to-the-rescue-17671379687f)

## Implementation Resources - implementations of BERT.

- [google-research/bert](https://github.com/google-research/bert)
The original code. TensorFlow code and pre-trained models for BERT.

- [huggingface/transformers](https://github.com/huggingface/transformers)
￼Transformers: State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch.

- [codertimo/BERT-pytorch](https://github.com/codertimo/BERT-pytorch)
Google AI 2018 BERT pytorch implementation

- [kaushaltrivedi/fast-bert](https://github.com/kaushaltrivedi/fast-bert)
Super easy library for BERT based NLP models

- [CyberZHG/keras-bert](https://github.com/CyberZHG/keras-bert)
Implementation of BERT that could load official pre-trained models for feature extraction and prediction

- [hanxiao/bert-as-service](https://github.com/hanxiao/bert-as-service)
 `bert-as-service` uses BERT as a sentence encoder and hosts it as a service via ZeroMQ, allowing you to map sentences into fixed-length representations in just two lines of code.


## Papers for Knowledge Distillation of Bert - reduce the model size and inference latency by distilling BERT into smaller models

- [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)

- [BERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)

- [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)

- [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)

- [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter (article)](https://medium.com/huggingface/distilbert-8cf3380435b5)
