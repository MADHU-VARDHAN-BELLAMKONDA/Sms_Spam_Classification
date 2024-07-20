import pandas as pd
import string

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])
    df['message'] = df['message'].apply(preprocess_text)
    return df

if __name__ == "__main__":
    file_path = "C:/Users/MADHU VARDHAN BK/Downloads/SMSSpamCollection"
    df = load_and_preprocess(file_path)
    print(df.head())
