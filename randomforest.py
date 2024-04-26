import re
import tools as tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump


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
        df = tools.open_df("balanced_text.csv")
        df = tools.add_label_categories(df)
        tools.plot_category(df)
        print(df.head())

        print("\nCleaning the text...")
        df['text'] = df['text'].apply(clean_text)

        print("\nSplitting the data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label'], test_size=0.2, random_state=42)

        print("\nVectorizing the text using TfidfVectorizer...")
        vectorizer = TfidfVectorizer(max_features=3500)
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        print("\nTraining RandomForestClassifier...\n")
        model = RandomForestClassifier(random_state=42, n_estimators=50)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        print(f'Accuracy Score: {accuracy_score(y_test, predictions)}')
        print("Classification Report: \n")
        print(f'{classification_report(y_test, predictions)}')

        print("\nSaving the model...")
        dump(vectorizer, "vectorizer.joblib")
        dump(model, "model.joblib")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
