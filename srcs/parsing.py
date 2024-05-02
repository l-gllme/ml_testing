import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def clean_tokens(tokens):
    """Clean the tokens by removing unnecessary characters"""

    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    return tokens


with open('data/CV.txt', 'r') as f:
    text = f.read()

nltk.download('punkt')

# sentences = nltk.sent_tokenize(text)
# for i, sentence in enumerate(sentences):
    # print(f"{i + 1}: {sentence}")


tokens = nltk.word_tokenize(text)

for i, token in enumerate(tokens):
    if i == 100:
        break
    print(f"{token} - ", end="")

tokens = clean_tokens(tokens)

print("\n\nAfter cleaning the tokens:\n")

for i, token in enumerate(tokens):
    if i == 100:
        break
    print(f"{token} - ", end="")