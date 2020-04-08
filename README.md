# Bert in Production

A small collection of resources on using BERT (https://arxiv.org/abs/1810.04805
) and related Language Models in production environments.


## Implementations
Implementations and production-ready tools related to BERT.

- [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) This library was recently open-sourced by Microsoft; it contains several model-specific optimisations including one for transformer models. It works by compiling a model's architecture through the Open Neural Network Exchange (ONNX) standard and optimising it for a platform's hardware.

- [google-research/bert](https://github.com/google-research/bert)
The original code. TensorFlow code and pre-trained models for BERT.

- [pytorch/fairseq](https://github.com/pytorch/fairseq)
Facebook AI Research Sequence-to-Sequence Toolkit written in Python. Contains the original code for RoBERTa.

- [google-research/google-research](https://github.com/google-research/google-research/tree/master/albert)
Google AI Research. Contains original code for Albert.

- [huggingface/transformers](https://github.com/huggingface/transformers) Transformers: State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch.

- [huggingface/tokenizers](https://github.com/huggingface/tokenizers) Fast State-of-the-Art Tokenizers optimized for Research and Production

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

## Descriptive Resources
Articles and papers describing how BERT works.

- [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

- [Sequence-to-sequence modeling with `NN.TRANSFORMER` and `TORCHTEXT`](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

- [Exploring BERT's Vocabulary](http://juditacs.github.io/2019/02/19/bert-tokenization-stats.html)

- [BERT to the rescue!](https://towardsdatascience.com/bert-to-the-rescue-17671379687f)

- [Pre-training BERT from scratch with cloud TPU](https://mc.ai/pre-training-bert-from-scratch-with-cloud-tpu/)

- [BERT Technology introduced in 3-minutes](https://towardsdatascience.com/bert-technology-introduced-in-3-minutes-2c2f9968268c)

- [BERT, RoBERTa, DistilBERT, XLNet — which one to use?](https://towardsdatascience.com/bert-roberta-distilbert-xlnet-which-one-to-use-3d5ab82ba5f8#e18a-828e5fc317c7)

- [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/)

## Deep Analysis

These papers do a deep analysis of the internals of BERT. Understanding the internals of a model can enable more efficient optimisations.

- [What Does BERT Look At? An Analysis of BERT's Attention](https://nlp.stanford.edu/pubs/clark2019what.pdf)

- [Investigating BERT’s Knowledge of Language: Five Analysis Methods with NPIs](https://arxiv.org/abs/1909.02597)

- [On Identifiability in Transformers](https://arxiv.org/abs/1908.04211)

- [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650)

## General Resources

Original papers describing architectures and methodologies intrisinc to a BERT-style language model.

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
Original BERT paper.

- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://www.aclweb.org/anthology/D18-2012.pdf)
Paper describing a similar sub-word tokenization approach to BERT's.

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) This paper introduced the Transformer architecture.

## Speed

One of the big problems with running BERT-like models in production is the time required to infer; a logical conclusion is that a faster model is a more production-ready model.

### Knowledge Distillation

One way to make a model faster is to reduce the amount of computation required to generate its output - Knowledge Distillation is the process of training a smaller "student" model from a larger "teacher" network. The smaller model is then deployed to production.

- [Small and Practical BERT Models for Sequence Labeling](https://arxiv.org/abs/1909.00100)

- [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)

- [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)

- [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) DistilBERT is a popular distilled bert by the authors of the huggingface library. They use half of the number of layers as BERT. [Related article](https://medium.com/huggingface/distilbert-8cf3380435b5)

### Compression

- [Small and Practical BERT Models for Sequence Labeling](https://arxiv.org/abs/1909.00100) Starting from a public multilingual BERT checkpoint, their final model is 6x smaller and 27x faster, and has higher accuracy than a state-of-the-art multilingual baseline.

- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) Albert primarily aims to reduce the number of trainable parameters in a BERT model. Albert shares all weights in the transformer encoder layers and decouples the dimension of the word embeddings from the dimensions of the transformer. The result is a model that has far fewer trainable parameters. Time to infer is not reduced.

- [Compression BERT for faster prediction](https://blog.rasa.com/compressing-bert-for-faster-prediction-2/)
Learn how to use pruning to speed up BERT.

- [Extreme Language Model Compression with Optimal Subwords and Shared Projections](https://arxiv.org/abs/1909.11687) The authors utilise knowledge distillation to train the teacher and student models simultaneously to obtain optimal word embeddings for the student vocabulary. Their method compresses the BERT_BASE model by more than 60x and to under 7MB. 

- [Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT](https://arxiv.org/abs/1909.05840) The authors propose a new quantization scheme and achieve comparable performance to baseline even with up to 13× compression of the model parameters and up to 4× compression of the embedding table as well as activations.

- [PoWER-BERT: Accelerating BERT inference for Classification Tasks](https://arxiv.org/abs/2001.08950) BERT has emerged as a popular model for natural language understanding. Given its compute intensive nature, even for inference, many recent studies have considered optimization of two important performance characteristics: model size and inference time. We consider classification tasks and propose a novel method, called PoWER-BERT, for improving the inference time for the BERT model without significant loss in the accuracy. The method works by eliminating word-vectors (intermediate vector outputs) from the encoder pipeline. We design a strategy for measuring the significance of the word-vectors based on the self-attention mechanism of the encoders which helps us identify the word-vectors to be eliminated. Experimental evaluation on the standard GLUE benchmark shows that PoWER-BERT achieves up to 4.5x reduction in inference time over BERT with < 1% loss in accuracy. We show that compared to the prior inference time reduction methods, PoWER-BERT offers better trade-off between accuracy and inference time. Lastly, we demonstrate that our scheme can also be used in conjunction with ALBERT (a highly compressed version of BERT) and can attain up to 6.8x factor reduction in inference time with < 1% loss in accuracy.

- [Q8BERT: Quantized 8Bit BERT](https://arxiv.org/abs/1910.06188) Recently, pre-trained Transformer based language models such as BERT and GPT, have shown great improvement in many Natural Language Processing (NLP) tasks. However, these models contain a large amount of parameters. The emergence of even larger and more accurate models such as GPT2 and Megatron, suggest a trend of large pre-trained Transformer models. However, using these large models in production environments is a complex task requiring a large amount of compute, memory and power resources. In this work we show how to perform quantization-aware training during the fine-tuning phase of BERT in order to compress BERT by 4× with minimal accuracy loss. Furthermore, the produced quantized model can accelerate inference speed if it is optimized for 8bit Integer supporting hardware. 

- [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351) Language model pre-training, such as BERT, has significantly improved the performances of many natural language processing tasks. However, pre-trained language models are usually computationally expensive and memory intensive, so it is difficult to effectively execute them on some resource-restricted devices. To accelerate inference and reduce model size while maintaining accuracy, we firstly propose a novel transformer distillation method that is a specially designed knowledge distillation (KD) method for transformer-based models. By leveraging this new KD method, the plenty of knowledge encoded in a large teacher BERT can be well transferred to a small student TinyBERT. Moreover, we introduce a new two-stage learning framework for TinyBERT, which performs transformer distillation at both the pre-training and task-specific learning stages. This framework ensures that TinyBERT can capture both the general-domain and task-specific knowledge of the teacher BERT.TinyBERT is empirically effective and achieves more than 96% the performance of teacher BERTBASE on GLUE benchmark while being 7.5x smaller and 9.4x faster on inference. TinyBERT is also significantly better than state-of-the-art baselines on BERT distillation, with only about 28% parameters and about 31% inference time of them. 

## Other Resources

- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)
Original RoBERTa paper.

- [Deploying BERT in production](https://towardsdatascience.com/deploy-bert-ef20636fc337)

- [Serving Google BERT in Production using Tensorflow and ZeroMQ](http://hanxiao.io/2019/01/02/Serving-Google-BERT-in-Production-using-Tensorflow-and-ZeroMQ/)

- [Pruning bert to accelerate inference](https://blog.rasa.com/pruning-bert-to-accelerate-inference/)
Learn how to make BERT smaller and faster

- [Improving Neural Machine Translation with Parent-Scaled Self-Attention](https://arxiv.org/abs/1909.03149)

- [Reducing Transformer Depth on Demand with Structured Dropout](https://arxiv.org/abs/1909.11556)
