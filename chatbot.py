"""FAQ Chatbot engine.

Pipeline: load FAQs -> preprocess (NLTK) -> TF-IDF vectors -> cosine similarity
-> return the answer of the most similar FAQ (or a fallback if confidence is low).

Run in terminal:  python chatbot.py
"""
import json
import re
from pathlib import Path

from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ_FILE = Path(__file__).parent / "faqs.json"
THRESHOLD = 0.12  # minimum similarity to accept a match

_tokenizer = RegexpTokenizer(r"[a-z0-9]+")  # no external downloads needed
_stemmer = PorterStemmer()
# keep a few words that carry meaning in FAQ questions
_STOP = set(ENGLISH_STOP_WORDS) - {"how", "what", "can", "do"}
_STOP |= {"how", "what", "do", "does", "can", "i", "my", "you", "your", "please", "product", "products", "want", "need"}


def preprocess(text: str) -> str:
    """Lowercase, remove punctuation, tokenize, drop stopwords, stem."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = _tokenizer.tokenize(text)
    tokens = [_stemmer.stem(t) for t in tokens if t not in _STOP]
    return " ".join(tokens)


class FAQChatbot:
    def __init__(self, faq_path=FAQ_FILE):
        with open(faq_path, encoding="utf-8") as f:
            self.faqs = json.load(f)
        # build one searchable document per FAQ
        # question (x2 for weight) + keywords/paraphrases + answer
        corpus = [preprocess(f["question"] + " " + f["question"] + " "
                             + f.get("keywords", "") + " " + f["answer"])
                  for f in self.faqs]
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(corpus)

    def get_response(self, user_input: str):
        """Return (answer, matched_question, score)."""
        cleaned = preprocess(user_input)
        if not cleaned:
            return ("Sorry, I couldn't understand that. Try rephrasing your question."), None, 0.0
        vec = self.vectorizer.transform([cleaned])
        scores = cosine_similarity(vec, self.matrix)[0]
        best = int(scores.argmax())
        score = float(scores[best])
        if score < THRESHOLD:
            return ("Sorry, I couldn't find an answer to that. Try rephrasing, "
                    "or contact support@shopeasy.example."), None, score
        return self.faqs[best]["answer"], self.faqs[best]["question"], score


def main():
    bot = FAQChatbot()
    print("ShopEasy FAQ Bot (type 'quit' to exit)")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", "bye"}:
            print("Bot: Goodbye!")
            break
        answer, matched, score = bot.get_response(q)
        print(f"Bot: {answer}")
        if matched:
            print(f"     [matched: '{matched}' | similarity {score:.2f}]")


if __name__ == "__main__":
    main()
