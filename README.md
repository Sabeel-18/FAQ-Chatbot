# FAQ Chatbot (ShopEasy)

A retrieval-based FAQ chatbot.

**Pipeline:** FAQs (`faqs.json`) → NLTK preprocessing (lowercase, regex tokenization, stop-word removal, Porter stemming) → TF-IDF vectors (scikit-learn) → cosine similarity → best-matching answer (fallback message if similarity < threshold).

## Run
```bash
pip install -r requirements.txt
python app.py        # web chat UI at http://127.0.0.1:5000
python chatbot.py    # terminal chatbot
```

## Files
- `faqs.json` – 20 FAQs (question, answer, keywords/paraphrases)
- `chatbot.py` – preprocessing + matching engine
- `app.py`, `templates/index.html` – Flask chat UI

To change the topic, just edit `faqs.json`.
