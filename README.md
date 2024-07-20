# SMS Spam Classification

This project classifies SMS messages as either 'spam' or 'ham' (not spam) using various machine learning models and deep learning techniques. The SMS Spam Collection dataset is used for training and evaluation.

## Project Overview

The project includes the following steps:
1. **Data Extraction:** Extracting the dataset from a ZIP file.
2. **Data Preprocessing:** Cleaning and preparing the data for modeling.
3. **Model Training:** Training and evaluating machine learning models.
4. **Deep Learning Model:** Building and training a deep learning model for classification.

## Project Structure

**Dataset**
The dataset used is the SMS Spam Collection dataset, which includes 5,574 SMS messages categorized as 'ham' (legitimate) or 'spam' (unsolicited messages). The dataset is provided in a text file where each line contains a label and a message separated by a tab.

**Project Structure**
The project is divided into separate tasks, each managed in its own repository for better organization:

**Data Extraction:** Extract the ZIP file containing the dataset.
**Data Preprocessing:** Process the text data by removing punctuation and converting text to lowercase.
**Model Training:** Train and evaluate machine learning models, including Multinomial Naive Bayes and Logistic Regression, and perform hyperparameter tuning.
**Deep Learning Model:** Tokenize text sequences, pad them, and build an LSTM-based deep learning model using Keras and TensorFlow.
## Usage

### Prerequisites

Ensure you have the necessary libraries installed:

```bash
pip install
 pandas
numpy
scikit-learn
tensorflow
