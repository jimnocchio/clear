# import nltk
# from nltk import pos_tag, ne_chunk, word_tokenize
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# sentence="Apple is built by Stevejobs"
# tokens=word_tokenize(sentence)
# tags=pos_tag(tokens)
# ner_tree=ne_chunk(tags)

# print(ner_tree)

# from textblob import TextBlob
# text="I very love this phone, The design is amazing and the battery life is great"
# blob=TextBlob(text)
# sentiment=blob.sentiment
# print(f"sentiment: {sentiment}")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

texts=["I love this movie","This is a great movie", "It was terrible"]
labels=[1,1,0]

texts_train, texts_test, labels_train, labels_test=train_test_split(texts, labels, test_size=0.2, random_state=42)

vectorizer=CountVectorizer()
X_train=vectorizer.fit_transform(texts_train)
X_test=vectorizer.transform(texts_test)

model=MultinomialNB()
model.fit(X_train, labels_train)

predictions=model.predict(X_test)

print("Accuracy: ", accuracy_score(labels_test,predictions))