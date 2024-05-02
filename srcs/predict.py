from joblib import load

label_names = {0: "sadness(0)", 1: "joy(1)", 2: "love(2)",
               3: "anger(3)", 4: "fear(4)", 5: "surprise(5)"}


def main():
    """Main function."""

    try:
        model = load("models/model.joblib")
        vectorizer = load("models/vectorizer.joblib")

        while True:
            text = input("\nEnter a text: ")
            if text.lower() == "exit":
                break

            prediction = model.predict(vectorizer.transform([text.lower()]))
            print(f"Prediction: {label_names[prediction[0]]}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()