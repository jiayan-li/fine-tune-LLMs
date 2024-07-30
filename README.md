# Fine-Tuning DistilBERT for Sentiment Analysis

This repository contains a Jupyter Notebook (`fine_tuning.ipynb`) for fine-tuning a pre-trained DistilBERT model to perform sentiment analysis on the IMDB dataset. The notebook provides step-by-step instructions for preparing the dataset, tokenizing the data, fine-tuning the model, evaluating its performance, and comparing it with a baseline model. The main goal of this project is to demonstrate how to leverage transfer learning to build a sentiment analysis model using a pre-trained language model. The main deep learning framework used in this project is PyTorch, and the Hugging Face Transformers library is used to work with the DistilBERT model.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Dataset Preparation](#dataset-preparation)
4. [Tokenization](#tokenization)
5. [Model Fine-Tuning](#model-fine-tuning)
6. [Evaluation](#evaluation)
7. [Inference](#inference)
8. [Baseline Comparison](#baseline-comparison)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction
This project demonstrates how to fine-tune a DistilBERT model for sentiment analysis using the IMDB dataset. DistilBERT is a smaller, faster, and lighter version of BERT, making it suitable for tasks requiring efficient computation while maintaining high accuracy.

## Installation
To run the notebook, you need to install the necessary libraries. You can do this by running the following command:

```bash
pip install transformers datasets torch numpy pandas scikit-learn
```

## Dataset Preparation
The IMDB dataset is used for training and evaluating the sentiment analysis model. The dataset contains movie reviews labeled as positive or negative. The notebook guides you through downloading and preparing the dataset for training.

## Tokenization
We use the `DistilBertTokenizer` from the `transformers` library to tokenize the dataset. Tokenization is the process of converting text into a format that can be used by the DistilBERT model. This step includes:
- Loading the tokenizer
- Tokenizing the text data
- Padding and truncating the tokenized sequences to ensure uniform input size

## Model Fine-Tuning
The notebook demonstrates how to fine-tune the DistilBERT model using the tokenized IMDB dataset. Fine-tuning involves:
- Initializing the pre-trained DistilBERT model
- Defining the training arguments and optimizer
- Training the model on the training dataset
- Saving the fine-tuned model for future use

## Evaluation
After fine-tuning, the model is evaluated on the test set to assess its performance. The evaluation metrics include accuracy, precision, recall, and F1-score.

## Inference
The notebook includes examples of how to perform inference using the fine-tuned model on custom sentences. This section demonstrates the model's ability to predict the sentiment of unseen text.

## Baseline Comparison
To highlight the improvement achieved by fine-tuning, the notebook compares the fine-tuned model's performance with a non-fine-tuned baseline model. This comparison helps illustrate the effectiveness of the fine-tuning process.

## Contributing
Contributions to this repository are welcome. If you have any ideas, suggestions, or improvements, please feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

By following the instructions in the notebook, you will learn how to fine-tune a DistilBERT model for sentiment analysis and understand the key steps involved in the process. This project serves as a practical example of applying transfer learning to a real-world NLP task.