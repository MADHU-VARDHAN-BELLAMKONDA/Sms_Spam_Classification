import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def train_and_evaluate(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['message'])
    y = df['label'].map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    param_grid = {'alpha': [0.1, 0.5, 1.0]}
    grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f'Best Parameters: {best_params}')

    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    best_f1 = f1_score(y_test, y_pred_best)
    print(f'Best F1 Score: {best_f1}')

    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    lr_f1 = f1_score(y_test, y_pred_lr)
    print(f'Logistic Regression F1 Score: {lr_f1}')

if __name__ == "__main__":
    file_path = "C:/Users/MADHU VARDHAN BK/Downloads/SMSSpamCollection"
    df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])
    df['message'] = df['message'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower())
    train_and_evaluate(df)
