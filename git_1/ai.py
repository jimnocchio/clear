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

# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score

# texts=["I love this movie","This is a great movie", "It was terrible"]
# labels=[1,1,0]

# texts_train, texts_test, labels_train, labels_test=train_test_split(texts, labels, test_size=0.2, random_state=42)

# vectorizer=CountVectorizer()
# X_train=vectorizer.fit_transform(texts_train)
# X_test=vectorizer.transform(texts_test)

# model=MultinomialNB()
# model.fit(X_train, labels_train)

# predictions=model.predict(X_test)

# print("Accuracy: ", accuracy_score(labels_test,predictions))


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 모델 생성
model = Sequential([
    # 입력층: 입력 데이터의 차원을 지정해야 합니다. 예를 들어, 입력 데이터가 20개의 특성을 가지고 있다면 input_shape=(20,)입니다.
    Dense(64, activation='relu', input_shape=(20,)),  
    # 드롭아웃 층: 학습 중 50%의 뉴런을 무작위로 비활성화하여 과적합 방지
    Dropout(0.5),
    # 은닉층: 추가적인 뉴런 층을 추가하여 모델의 학습 능력을 높입니다.
    Dense(64, activation='relu'),
    # 드롭아웃 층
    Dropout(0.5),
    # 출력층: 최종적으로 분류할 클래스가 2개(이진 분류)인 경우, 1개의 뉴런과 시그모이드 활성화 함수를 사용합니다.
    Dense(1, activation='sigmoid')
])

# 모델 컴파일: 옵티마이저, 손실 함수, 평가 지표를 설정합니다.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 조기 종료 설정: 검증 데이터의 손실이 2번의 에폭 동안 개선되지 않으면 학습을 조기에 종료
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# 모델 학습: 학습 데이터와 레이블, 에폭 수, 검증 데이터를 지정합니다.
# 여기서는 에폭 수를 20으로 설정했습니다. 실제 데이터셋에 따라 조정해야 합니다.
model.fit(train_data, train_labels, epochs=20, validation_data=(validation_data, validation_labels), callbacks=[early_stopping])
