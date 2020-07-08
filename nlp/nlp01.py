import pandas as pd
import pandas_profiling
# data = pd.read_csv('data\kaggle\spam\spam.csv',encoding='latin1')
data = pd.read_csv('d://private/data\kaggle\spam\spam.csv',encoding='latin1')

data.head(10)

pr=data.profile_report()

pr

####
# corpus - 자연어 데이터를 말뭉치 또는 코퍼스


#########문장별로 자르기##########

from nltk.tokenize import sent_tokenize
text="His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to mae sure no one was near."
print(sent_tokenize(text))


# ['His barber kept his word.', 'But keeping such a huge secret to himself was driving him crazy.',
#  'Finally, the barber went up a mountain and almost to the edge of a cliff.', 'He dug a hole in the midst of some reeds.', 'He looked about, to mae sure no one was near.']

from nltk.tokenize import sent_tokenize
text="I am actively looking for Ph.D. students. and you are a Ph.D student."
print(sent_tokenize(text))

# ['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']

import kss
text='딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어려워요. 농담아니에요. 이제 해보면 알걸요?'
print(kss.split_sentences(text))

# ['딥 러닝 자연어 처리가 재미있기는 합니다.', '그런데 문제는 영어보다 한국어로 할 때 너무 어려워요.', '농담아니에요.', '이제 해보면 알걸요?']


##########단어별로 자르기############

from nltk.tokenize import word_tokenize
text="I am actively looking for Ph.D. students. and you are a Ph.D. student."
print(word_tokenize(text))

# ['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']

from nltk.tag import pos_tag
x=word_tokenize(text)
pos_tag(x)

# [('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'),
#  ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), ('student', 'NN'), ('.', '.')]

# 영어 문장에 대해서 토큰화를 수행하고, 이어서 품사 태깅을 수행하였습니다. 
# Penn Treebank POG Tags에서 PRP는 인칭 대명사, VBP는 동사, RB는 부사, 
# VBG는 현재부사, IN은 전치사, NNP는 고유 명사, NNS는 복수형 명사, CC는 접속사, DT는 관사를 의미합니다.

#기태 왈, mecab이 설치는 어렵지만 성능이 괜찮다.

# import konlpy

from konlpy.tag import Okt  
okt=Okt()  
print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))

# ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']  

print(okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  

# [('열심히','Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), 
# ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]  

print(okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  

# ['코딩', '당신', '연휴', '여행']  

# 1) morphs : 형태소 추출
# 2) pos : 품사 태깅(Part-of-speech tagging)
# 3) nouns : 명사 추출


#mecab - 설치링크
# https://cleancode-ws.tistory.com/97




###################################################################################


# 1) 어간(stem)
# : 단어의 의미를 담고 있는 단어의 핵심 부분.

# 2) 접사(affix)
# : 단어에 추가적인 의미를 주는 부분.

# 포터 알고리즘(Porter Algorithm)

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
s = PorterStemmer()
text="This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
words=word_tokenize(text)
print(words)

# ['This', 'was', 'not', 'the', 'map', 'we', 'found', 'in', 'Billy', 'Bones', "'s", 'chest', ',', 'but', 'an', 
# 'accurate', 'copy', ',', 'complete', 'in', 'all', 'things', '--', 'names', 'and',
#  'heights', 'and', 'soundings', '--', 'with', 'the', 'single', 'exception', 'of', 'the', 'red', 'crosses', 'and', 'the', 'written', 'notes', '.']

from nltk.stem import PorterStemmer
s=PorterStemmer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([s.stem(w) for w in words])
# ['polici', 'do', 'organ', 'have', 'go', 'love', 'live', 'fli', 'die', 'watch', 'ha', 'start']

from nltk.stem import LancasterStemmer
l=LancasterStemmer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([l.stem(w) for w in words])
# ['policy', 'doing', 'org', 'hav', 'going', 'lov', 'liv', 'fly', 'die', 'watch', 'has', 'start']

# Stemming
# am → am
# the going → the go
# having → hav

# Lemmatization
# am → be
# the going → the going
# having → have


# 1. NLTK에서 불용어 확인하기

# import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
stopwords.words('english')[:10]
# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your'] 


# 2. NLTK를 통해서 불용어 제거하기
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english')) 

word_tokens = word_tokenize(example)

result = []
for w in word_tokens: 
    if w not in stop_words: 
        result.append(w) 

print(word_tokens) 
print(result) 
# ['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
# ['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
