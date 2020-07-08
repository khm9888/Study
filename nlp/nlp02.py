# 추후 할 것.
# 웹 크롤링으로 데이터 긁어오기

# 지금 할 것
# word_tokenize로 단어별로 구분
# Counter를 통해서 모으고 순서대로 most_common 할 것
from keras.preprocessing.text import Tokenizer

docs = ["너무 재밋어요", "참 최고에요", "참 잘 만든 영화에요","추천하고 싶은 영화입니다", "한 번 더 보고 싶네요", 
        "글쎄요", "별로에요", "생각보다 지루해요", "연기가 어색해요", "재미없어요",
        "너무 재미없다", "참 재밋네요", "재미없어요"]

# 긍정 1, 부정0
labels = np.array([1,1,1,1,1,1,0,0,0,0,0,0,0])
# labels = []

# 토큰화 
token = Tokenizer()
token.fit_on_texts(docs)# 각 워드별로 잘라서, 빈도 수 순으로 학습
print("token.word_index")
print(token.word_index)# 빈도 순으로 정렬해서, index 한다.

# token.word_index
# {'참': 1, '너무': 2, '재미없어요': 3, '재밋어요': 4, '최고에요': 5, '잘': 6, '만든': 7, '영화에요': 8, '추천하고': 9, '싶은': 10, '영화입니다': 11, '한번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로에요': 17, '생각보
# 다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없다': 22, '재밋네요': 23


x = token.texts_to_sequences(docs)
print("texts_to_sequences")
print(x)



# texts_to_sequences
# [[2, 4], [1, 5], [1, 6, 7, 8], [9, 10, 11], [12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [3], [2, 22], [1, 23], [3]]

token.

print(token.fit_on_texts)