from keras.preprocessing.text import Tokenizer,on
from keras.utils.np_utils import to_categorical
text = "i eat a delicious rice"

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)#잘라서, 각 워드를 key, 그리고 순서를 value로 한다.
#{'i': 1, 'eat': 2, 'a': 3, 'delicious': 4, 'rice': 5}

x = token.texts_to_sequences([text])

print(x)
#[[1, 2, 3, 4, 5]]

word_size = len(token.word_index)+1
x = to_categorical(x,num_classes=word_size)
x = to_categorical(x)
print(x)