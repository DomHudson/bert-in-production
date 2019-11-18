# Bert in Production

A small collection of resources on using BERT (https://arxiv.org/abs/1810.04805
) and related Language Models in production environments.


## Descriptive Resources - how BERT works.

- [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

- [Sequence-to-sequence modeling with `NN.TRANSFORMER` and `TORCHTEXT`](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

- [Exploring BERT's Vocabulary](http://juditacs.github.io/2019/02/19/bert-tokenization-stats.html)

- [BERT to the rescue!](https://towardsdatascience.com/bert-to-the-rescue-17671379687f)

- [Pre-training BERT from scratch with cloud TPU](https://mc.ai/pre-training-bert-from-scratch-with-cloud-tpu/)

- [BERT Technology introduced in 3-minutes](https://towardsdatascience.com/bert-technology-introduced-in-3-minutes-2c2f9968268c)

- [BERT, RoBERTa, DistilBERT, XLNet — which one to use?](https://towardsdatascience.com/bert-roberta-distilbert-xlnet-which-one-to-use-3d5ab82ba5f8#e18a-828e5fc317c7)

- [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/)

## Implementation Resources - implementations of BERT.

- [google-research/bert](https://github.com/google-research/bert)
The original code. TensorFlow code and pre-trained models for BERT.

- [pytorch/fairseq](https://github.com/pytorch/fairseq)
Facebook AI Research Sequence-to-Sequence Toolkit written in Python. Contains the original code for RoBERTa.

- [google-research/google-research](https://github.com/google-research/google-research/tree/master/albert)
Google AI Research. Contains original code for Albert.

- [huggingface/transformers](https://github.com/huggingface/transformers)
￼Transformers: State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch.

- [spacy-transformers](https://github.com/explosion/spacy-transformers)
spaCy pipelines for pre-trained BERT, XLNet and GPT-2 

- [codertimo/BERT-pytorch](https://github.com/codertimo/BERT-pytorch)
Google AI 2018 BERT pytorch implementation

- [kaushaltrivedi/fast-bert](https://github.com/kaushaltrivedi/fast-bert)
Super easy library for BERT based NLP models

- [CyberZHG/keras-bert](https://github.com/CyberZHG/keras-bert)
Implementation of BERT that could load official pre-trained models for feature extraction and prediction

- [hanxiao/bert-as-service](https://github.com/hanxiao/bert-as-service)
 `bert-as-service` uses BERT as a sentence encoder and hosts it as a service via ZeroMQ, allowing you to map sentences into fixed-length representations in just two lines of code.

## General Resources

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
Original BERT paper.

- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)
Original RoBERTa paper.

- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://www.aclweb.org/anthology/D18-2012.pdf)
Paper describing a similar sub-word tokenization approach to BERT's.

## Knowledge Distillation

One technique to make large models more production ready is to train a smaller "student" network on the outputs of the larger "teacher" network. This is called Knowledge Distillation.

- [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)

- [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)

- [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) DistilBERT is a popular distilled bert by the authors of the huggingface library. They use half of the number of layers as BERT. [Related article](https://medium.com/huggingface/distilbert-8cf3380435b5)

## Other Resources

- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) Albert primarily aims to reduce the number of trainable parameters in a BERT model. Albert shares all weights in the transformer encoder layers and decouples the dimension of the word embeddings from the dimensions of the transformer. 

- [Deploying BERT in production](https://towardsdatascience.com/deploy-bert-ef20636fc337)

- [Serving Google BERT in Production using Tensorflow and ZeroMQ](http://hanxiao.io/2019/01/02/Serving-Google-BERT-in-Production-using-Tensorflow-and-ZeroMQ/)

- [Pruning bert to accelerate inference](https://blog.rasa.com/pruning-bert-to-accelerate-inference/)
Learn how to make BERT smaller and faster

- [Compression BERT for faster prediction](https://blog.rasa.com/compressing-bert-for-faster-prediction-2/)
Learn how to use pruning to speed up BERT

- [Extreme Language Model Compression with Optimal Subwords and Shared Projections](https://arxiv.org/abs/1909.11687)

- [Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT](https://arxiv.org/abs/1909.05840)

- [Improving Neural Machine Translation with Parent-Scaled Self-Attention](https://arxiv.org/abs/1909.03149)

- [Reducing Transformer Depth on Demand with Structured Dropout](https://arxiv.org/abs/1909.11556)
