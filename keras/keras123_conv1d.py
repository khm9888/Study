from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재밋어요", "참 최고에요", "참 잘 만든 영화에요","추천하고 싶은 영화입니다", "한 번 더 보고 싶네요", 
        "글쎄요", "별로에요", "생각보다 지루해요", "연기가 어색해요", "재미없어요",
        "너무 재미없다", "참 재밋네요", "재미없어요"]

# 긍정 1, 부정0
labels = np.array([1,1,1,1,1,1,0,0,0,0,0,0,0])
# labels = []

# 토큰화 
token = Tokenizer()
token.fit_on_texts(docs)
print("token.word_index")
print(token.word_index)

# token.word_index
# {'참': 1, '너무': 2, '재미없어요': 3, '재밋어요': 4, '최고에요': 5, '잘': 6, '만든': 7, '영화에요': 8, '추천하고': 9, '싶은': 10, '영화입니다': 11, '한번': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로에요': 17, '생각보
# 다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없다': 22, '재밋네요': 23}

x = token.texts_to_sequences(docs)
print("texts_to_sequences")
print(x)

# texts_to_sequences
# [[2, 4], [1, 5], [1, 6, 7, 8], [9, 10, 11], [12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [3], [2, 22], [1, 23], [3]]

from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre') # ex )0 이 앞에서 채워짐 0 0 0 3 7
print("pad_x")
print(pad_x)

''' pad_x
[[ 0  0  2  4]
 [ 0  0  1  5]
 [ 1  6  7  8]
 [ 0  9 10 11]
 [12 13 14 15]
 [ 0  0  0 16]
 [ 0  0  0 17]
 [ 0  0 18 19]
 [ 0  0 20 21]
 [ 0  0  0  3]
 [ 0  0  2 22]
 [ 0  0  1 23]
 [ 0  0  0  3]] '''

pad_y = pad_sequences(x, padding='post', value=1.0) # ex )0 이 뒤에서 채워짐 3 7 0 0 0 / value = 1  0대신 1이 채워짐
print("pad_y")
print(pad_y)

''' pad_y
[[ 2  4  1  1]
 [ 1  5  1  1]
 [ 1  6  7  8]
 [ 9 10 11  1]
 [12 13 14 15]
 [16  1  1  1]
 [17  1  1  1]
 [18 19  1  1]
 [20 21  1  1]
 [ 3  1  1  1]
 [ 2 22  1  1]
 [ 1 23  1  1]
 [ 3  1  1  1]]
 '''

#######################################
# 문자 수정했을 때

'''token.word_index
{'참': 1, '너무': 2, '재미없어요': 3, '재밋어요': 4, '최고에요': 5, '잘': 6, '만든': 7, '영화에요': 8, '추천하고': 9, '싶은': 10, '영화입니다': 11, '한': 12, '번': 13, '더': 14, '보고': 15, '싶네요': 16, '글쎄요': 17, '별로에요': 18, 
'생각보다': 19, '지루해요': 20, '연기가': 21, '어색해요': 22, '재미없다': 23, '재밋네요': 24}
texts_to_sequences
[[2, 4], [1, 5], [1, 6, 7, 8], [9, 10, 11], [12, 13, 14, 15, 16], [17], [18], [19, 20], [21, 22], [3], [2, 23], [1, 24], [3]]
pad_x
 [ 0  0  0  1  5]
 [ 0  1  6  7  8]
 [ 0  0  9 10 11]
 [12 13 14 15 16]
 [ 0  0  0  0 17]
 [ 0  0  0  0 18]
 [ 0  0  0 19 20]
 [ 0  0  0 21 22]
 [ 0  0  0  0  3]
 [ 0  0  0  2 23]
 [ 0  0  0  1 24]
 [ 0  0  0  0  3]]
pad_y
[[ 2  4  1  1  1]
 [ 1  5  1  1  1]
 [ 1  6  7  8  1]
 [ 9 10 11  1  1]
 [12 13 14 15 16]
 [17  1  1  1  1]
 [18  1  1  1  1]
 [19 20  1  1  1]
 [21 22  1  1  1]
 [ 3  1  1  1  1]
 [ 2 23  1  1  1]
 [ 1 24  1  1  1]
 [ 3  1  1  1  1]]'''

print(pad_x.shape)

word_size = len(token.word_index)+1
print(word_size)

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten,Conv1D


# #dnn
# model = Sequential()
# model.add(Embedding(word_size,10,input_length=5))
# model.add(Flatten())
# model.add(Dense(1,activation="sigmoid"))
pad_x = np.array(pad_x)
# print(type(pad_x))
pad_x=pad_x.reshape(-1,pad_x.shape[1],1)

#rnn
model = Sequential()
# model.add(Embedding(25,10))
model.add(Conv1D(20,2,activation="relu",input_shape=(5,1)))
model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))

model.summary()

#훈련

model.compile(loss ="categorical_crossentropy", optimizer="adam",metrics=["acc"])
model.fit(pad_x,labels,epochs=10)

acc=model.evaluate(pad_x,labels)[1]
print(acc)