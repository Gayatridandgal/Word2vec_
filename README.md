# Word2Vec Applications: Dot Product, Cosine Similarity, and Sentiment Analysis

This repository contains a Jupyter notebook with three different programs demonstrating the use of Word2Vec embeddings:
1. Calculating the dot product between two words using Word2Vec embeddings.
2. Calculating the cosine similarity between two words using Word2Vec embeddings.
3. Performing sentiment analysis on sample comments (positive/negative) using Word2Vec embeddings and a simple two-layer neural network.

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Usage](#usage)
    - [Dot Product Calculation](#dot-product-calculation)
    - [Cosine Similarity Calculation](#cosine-similarity-calculation)
    - [Sentiment Analysis](#sentiment-analysis)
4. [Conclusion](#conclusion)

## Introduction

Word2Vec is a popular technique for natural language processing that creates dense vector representations for words based on their context in a corpus of text. This notebook demonstrates three distinct applications of Word2Vec embeddings:
- **Dot Product**: Measures the similarity between two word vectors.
- **Cosine Similarity**: Measures the cosine of the angle between two word vectors.
- **Sentiment Analysis**: Uses word vectors and a neural network to classify the sentiment of text as positive or negative.

## Requirements

To run the notebook, you need the following libraries installed:

- `gensim`
- `numpy`
- `tensorflow`
- `scikit-learn`

You can install the required libraries using pip:

```bash
pip install gensim numpy tensorflow scikit-learn
```

## Usage

### Dot Product Calculation

This program calculates the dot product between two words using their Word2Vec embeddings.

1. **Load Pre-trained Word2Vec Model**: Load a pre-trained Word2Vec model (e.g., Google News vectors).
2. **Get Word Vectors**: Retrieve the vectors for the two words from the model.
3. **Compute Dot Product**: Calculate the dot product between the two word vectors.

### Cosine Similarity Calculation

This program calculates the cosine similarity between two words using their Word2Vec embeddings.

1. **Load Pre-trained Word2Vec Model**: Load a pre-trained Word2Vec model.
2. **Get Word Vectors**: Retrieve the vectors for the two words from the model.
3. **Compute Cosine Similarity**: Use a function to calculate the cosine similarity between the two word vectors.

### Sentiment Analysis

This program performs sentiment analysis on sample comments using Word2Vec embeddings and a simple two-layer neural network.

1. **Load Pre-trained Word2Vec Model**: Load a pre-trained Word2Vec model.
2. **Get Sentence Vectors**: Calculate the average word vectors for each sentence.
3. **Prepare Data**: Split the data into training and testing sets.
4. **Build Neural Network**: Create a two-layer neural network.
5. **Train Neural Network**: Train the neural network on the training data.
6. **Predict Sentiment**: Predict the sentiment of user-provided sentences.

## Conclusion

These three programs demonstrate different ways to leverage Word2Vec embeddings for natural language processing tasks. The dot product and cosine similarity provide measures of word similarity, while the sentiment analysis example shows how to use word vectors and a neural network for text classification.

Feel free to explore the notebook and modify the code for your own applications!

---
