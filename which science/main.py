import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from flask import Flask, request, jsonify
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# 1. Create a small dataset for training
data = {
    "text": [
        "Newton's laws of motion describe the relationship between the motion of an object and the forces acting on it.",
        "The chemical reaction between hydrogen and oxygen produces water.",
        "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize food.",
        "Quantum mechanics studies the behavior of matter and energy on the atomic and subatomic level.",
        "The periodic table organizes elements according to their properties.",
        "Genetics is the study of heredity and the variation of inherited characteristics.",
        "This article discusses the latest trends in modern art.",
        "Economics is the study of how people allocate scarce resources.",
        "Biology is the natural science that studies life and living organisms.",
        "Chemical bonds form when atoms share or transfer electrons.",
        "Physics explains the fundamental principles governing the universe.",
        "The theory of evolution by natural selection explains how species evolve over time.",
        "This text does not relate to science at all.",
    ],
    "label": [
        "Physics", "Chemistry", "Biology", "Physics", "Chemistry", "Biology", 
        "None", "None", "Biology", "Chemistry", "Physics", "Biology", "None"
    ]
}

df = pd.DataFrame(data)

# 2. Preprocess data and split into train and test sets
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Preprocess text by removing stopwords."""
    return ' '.join([word for word in text.lower().split() if word not in stop_words])

df['text'] = df['text'].apply(preprocess_text)

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create a text classification pipeline
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()), # Convert text to numerical data
    ('classifier', LogisticRegression()) # Logistic Regression for classification
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = model_pipeline.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# 4. Deploy the model using Flask
app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify_text():
    """
    API endpoint to classify input text.
    Input: JSON with 'text' field.
    Output: JSON with 'category' field.
    """
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    input_text = preprocess_text(data['text'])
    prediction = model_pipeline.predict([input_text])[0]
    return jsonify({"category": prediction})

if __name__ == '__main__':
    # Run Flask app locally
    app.run(debug=True)