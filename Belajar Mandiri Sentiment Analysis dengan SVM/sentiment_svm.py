from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Dataset sederhana
texts = [
    "I love this product",
    "This is terrible",
    "Very happy with the service",
    "I hate this thing",
    "Excellent job by the team",
    "Worst experience ever",
    "Absolutely fantastic",
    "Really bad and disappointing"
]

labels = [
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative"
]

# Buat pipeline: TF-IDF + SVM
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC(kernel='linear'))
])

# Latih model
model.fit(texts, labels)

# Uji coba
test_sentences = [
    "I am very happy today",
    "This is the worst product",
    "I really love it",
    "Terrible and horrible experience",
    "Absolutely fantastic"
]

# Cetak hasil prediksi
for sentence in test_sentences:
    prediction = model.predict([sentence])[0]
    print(f"'{sentence}' => {prediction}")
