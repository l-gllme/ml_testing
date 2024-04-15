import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import re
import tools as tools

def clean_text(text):
    """Clean the text by removing unnecessary characters"""
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    return text

def main():
    """Main function."""
    try:
        df = tools.open_df("text.csv")
        df = tools.add_label_categories(df)
        tools.plot_category(df)

        print("\nCleaning the text...")
        df['text'] = df['text'].apply(clean_text)
        print(df['text'].head())

        print("\nSplitting the data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label'], test_size=0.2, random_state=42)

        print("\nVectorizing the text using TfidfVectorizer...")
        vectorizer = TfidfVectorizer(max_features=3500)
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        print("\nTraining LinearSVC...\n")
        model = LinearSVC(random_state=42, dual=False)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        print(f'Accuracy Score: {accuracy_score(y_test, predictions)}')
        print(f'Classification Report: \n{classification_report(y_test, predictions)}')

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
