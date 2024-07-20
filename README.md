# SMS Spam Classification

This project aims to build a predictive model to classify SMS messages as either 'spam' or 'ham' (not spam) using various machine learning techniques in Python. The project is structured into multiple modular files for better organization and maintainability.

## Project Structure

### Files

1. **extract_data.py**
   - Contains the function `extract_data` which extracts the SMS Spam Collection dataset from a ZIP file.
   - The path to the ZIP file and the extraction directory are provided as arguments.

2. **preprocess_data.py**
   - Contains the function `preprocess_data` which handles data cleaning and text preprocessing.
   - Removes punctuation, converts text to lowercase, and splits the data into training and testing sets.

3. **train_model.py**
   - Contains the function `train_and_evaluate_model` which trains and evaluates machine learning models.
   - Uses TF-IDF vectorization, trains Multinomial Naive Bayes and Logistic Regression models, performs hyperparameter tuning, and evaluates performance metrics.

4. **deep_learning_model.py**
   - Contains the function `build_and_evaluate_dl_model` which builds and trains a deep learning model using LSTM.
   - Tokenizes and pads text sequences, builds an LSTM-based model, trains the model, and evaluates its performance.

5. **main.py**
   - The main script that integrates all the modules.
   - Extracts the dataset, preprocesses the data, trains and evaluates machine learning models, and builds and evaluates the deep learning model.
   - Paths to the dataset files and ZIP file are hardcoded in this script and should be adjusted as necessary.

## Steps in the Project

1. **Extract the Dataset:**
   - The dataset is extracted from a ZIP file using the `extract_data` function.

2. **Data Preprocessing:**
   - Text data is cleaned by removing punctuation and converting text to lowercase.
   - The data is split into training and testing sets.

3. **Model Training and Evaluation:**
   - **Machine Learning Models:**
     - TF-IDF vectorization is used to transform the text data into numerical features.
     - Multinomial Naive Bayes and Logistic Regression models are trained.
     - Hyperparameter tuning is performed for the Naive Bayes model using GridSearchCV.
     - Models are evaluated using accuracy, precision, recall, and F1 score.
   - **Deep Learning Model:**
     - Text sequences are tokenized and padded.
     - An LSTM-based deep learning model is built and trained using Keras and TensorFlow.
     - The model's performance is evaluated using accuracy.

## Usage

### Prerequisites

Ensure you have the necessary libraries installed:

```bash
pip install pandas numpy scikit-learn tensorflow
