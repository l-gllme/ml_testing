import re
import tools as tools
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
import numpy as np
from tqdm import tqdm

def clean_text(text):
    """Clean the text by removing unnecessary characters"""
    text = text.lower()
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    return text

def main():
    """Main function."""
    try:
        df = tools.open_df("data/text.csv")
        df = tools.add_label_categories(df)
        tools.plot_category(df)

        print("\nCleaning the text...")
        df['text'] = df['text'].apply(clean_text)

        print("\nSplitting the data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label'], test_size=0.2, random_state=42)

        print("\nVectorizing the text using TfidfVectorizer...")
        vectorizer = TfidfVectorizer(max_features=3500)
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        print("\nTraining SGDClassifier with hinge loss...")
        model = SGDClassifier(loss='hinge', max_iter=1, random_state=42, verbose=0)
        
        with tqdm(total=100, desc="Training SGDClassifier") as pbar:
            for _ in range(100):
                model.partial_fit(X_train, y_train, classes=np.unique(y_train))
                pbar.update(1)

        predictions = model.predict(X_test)
        print(f'Accuracy Score: {accuracy_score(y_test, predictions)}')
        print("Classification Report: \n")
        print(f'{classification_report(y_test, predictions)}')

        print("\nSaving the model...")
        dump(vectorizer, "models/vectorizer.joblib")
        dump(model, "models/model.joblib")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
