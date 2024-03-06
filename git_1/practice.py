import nltk
nltk.download('averaged_perceptron_tagger') #태깅을 사용하기 위함
nltk.download('punkt') #토큰화하기 위함

text= "Could you repeat that please?"
tokens=nltk.word_tokenize(text) #변수에 word_tokenize 함수를 통해 text를 토큰화함
print("Tokens:", tokens) 

tagged=nltk.pos_tag(tokens) #품사를 태그해주는 함수 pos_tag
print("Tagged Tokens",tagged)

#1. 형태소 분석: (stemming, Lemmatization)
#stemming:단어에서 접사를 제거하여 기본 형태를 찾는 방법이다. ex) running에서 run을 찾음
#PorterStemmer()을 사용한다.

#Lemmatization: 품사 정보를 활용해 단어의 사전적 혹은 의미적 기본형태를 찾는다. Stemming보다 더 정교한 방법이다.
#WordNetLemmatizer()을 사용한다.


#2. 구문 분석: 문장의 구조를 분석하여 명사구,동사구 등을 식별하는 과정이다.
#.

#3. 개체명 인식: 인명,지명,날짜 등과 같은 특정 정보를 식별함.
#ne chunk함수를 통해 NER작업을 수행가능
#form nltk import ne_chunk
#nltk.download('maxent_ne_chunker') 해야함

#4. 빈도 분석과 텍스트 분류: FreqDist 클래스를 사용해 텍스트 내 단어의 빈도를 분석할 수 있고,
#Naive Bayes알고리즘을 활용해 텍스트 데이터를 분류할 수 있어.
#from nltk import FreqDist