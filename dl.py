import re
import tools as tools
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

def clean_text(text):
    """Clean the text by removing unnecessary characters."""
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

        print("\nPreparing the data for the deep learning model...")
        # Tokenization and sequence padding
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(df['text'])
        sequences = tokenizer.texts_to_sequences(df['text'])
        X = pad_sequences(sequences, maxlen=200)
        y = np.array(df['label'])

        print("\nSplitting the data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("\nBuilding the deep learning model...")
        model = Sequential()
        model.add(Embedding(5000, 20, input_length=200))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # Use 'softmax' if you have more than two classes

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Use 'categorical_crossentropy' for more than two classes

        print("\nTraining the model...")
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=2)

        predictions = model.predict(X_test) > 0.5
        print(f'Accuracy Score: {accuracy_score(y_test, predictions)}')
        print("Classification Report: \n")
        print(f'{classification_report(y_test, predictions)}')

        print("\nSaving the model...")
        model.save("text_classification_model.h5")
        dump(tokenizer, "tokenizer.joblib")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
