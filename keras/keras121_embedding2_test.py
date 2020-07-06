from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
import numpy as np
from urllib.request import urlopen
import bs4

q_num=13136
url = f"https://www.acmicpc.net/problem/{q_num}"
source = urlopen(url).read()
source_bs4 = bs4.BeautifulSoup(source,"html.parser")
text_0=source_bs4.find('div', id ="problem_description").find_all('p')
text = [w for t in text_0 for w in t.string.split(".")]
# text = text[:len(text)//2]
print("text")
print(text)

labels = np.array([1,1,1,1,1,1,1,0,0,0,0,0,0,0])

token = Tokenizer()
token.fit_on_texts([text])

print("token.word_index")
print(token.word_index)
# {'acm': 1, 'icpc': 2, '대회의': 3, '대회장은': 4, 'r행': 5, 'c열의': 6, '직사각형': 7, '형태로': 8, 
# '좌석이': 9, '배치되어': 10, '있다': 11, '대회가': 12, '시작하기': 13, '전에는': 14, '참가자들이': 15, 
# '아무것도': 16, '만지면': 17, '안': 18, '되기': 19, '때문에': 20, '진행자는': 21, "'do": 22, 'not': 23, 
# 'touch': 24, 'anything': 25, "'을": 26, '연신': 27, '외친다': 28}

x = token.texts_to_sequences([text])
print("texts_to_sequences")
print(x)

# [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]

# word_size = len(token.word_index)+1
# x = to_categorical(x,num_classes=word_size)
x = to_categorical(x)
print("to_categorical")
print(x)


from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre') # ex )0 이 앞에서 채워짐 0 0 0 3 7
print(pad_x)


pad_y = pad_sequences(x, padding='post', value=1.0) # ex )0 이 뒤에서 채워짐 3 7 0 0 0 / value = 1  0대신 1이 채워짐
print(pad_y)