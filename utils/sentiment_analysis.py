from textblob import TextBlob

def analyze_sentiment(text: str) -> float:
    blob = TextBlob(text)
    return blob.sentiment.polarity