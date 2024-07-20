import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def build_and_evaluate(df):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['message'])
    X_seq = tokenizer.texts_to_sequences(df['message'])
    X_pad = pad_sequences(X_seq, maxlen=100)

    X_train, X_test, y_train, y_test = train_test_split(X_pad, df['label'].map({'ham': 0, 'spam': 1}), test_size=0.2, random_state=42)

    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=100),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Deep Learning Model Accuracy: {accuracy}')

if __name__ == "__main__":
    file_path = "C:/Users/MADHU VARDHAN BK/Downloads/SMSSpamCollection"
    df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])
    df['message'] = df['message'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower())
    build_and_evaluate(df)
