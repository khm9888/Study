#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 예: x^2를 출력하는 함수 pow1(x)입니다
def pow1(x):
    return x ** 2


# In[2]:


# pow1(x)와 동일한 기능을 하는 익명 함수 pow2입니다
pow2 = lambda x: x ** 2


# In[5]:


# pow2에 인수 a를 전달하여, 계산 결과를 b에 저장합니다
# b = pow2(a)


# In[6]:


# 대입할 인수 a입니다
a = 4

# def를 이용하여 func1를 작성하세요
def func1(x):
    return 2 * x**2 - 3*x + 1

# lambda를 이용하여 func2를 작성하세요
func2 = lambda x: 2 * x**2 - 3*x + 1

# 반환값을 출력합니다
print(func1(a))
print(func2(a))


# In[7]:


# 예: 두 인수를 더하는 함수 add1입니다
add1 = lambda x, y: x + y


# In[8]:


print((lambda x, y: x + y)(3, 5))


# In[9]:


# 대입할 인수 x, y, z입니다
x = 5
y = 6
z = 2

# def를 이용하여 func3를 작성하세요
def func3(x, y, z):
    return x*y + z

# lambda를 이용하여 func4를 작성하세요
func4 = lambda x, y, z: x*y + z

# 출력합니다
print(func3(x, y, z))
print(func4(x, y, z))


# In[10]:


# "hello"를 출력하는 함수입니다
def say_hello():
    print("hello")


# In[11]:


# 인수가 3 미만이면 2를 곱하고, 3 이상이면 3으로 나누어 5를 더하는 함수입니다
def lower_three1(x):
    if x < 3:
        return x * 2
    else:
        return x/3 + 5


# In[13]:


# lower_three1와 동일한 함수
lower_three2 = lambda x: x * 2 if x < 3 else x/3 + 5


# In[15]:


# 대입할 인수 a1, a2입니다
a1 = 13
a2 = 32

# lambda를 이용하여 func5를 작성하세요
func5 = lambda x: x**2 - 40*x + 350 if x >= 10 and x < 30 else 50

# 반환값을 출력합니다
print(func5(a1))
print(func5(a2))


# In[17]:


# 나눌 문자열입니다
test_sentence = "this is a test sentence."

# split으로 리스트를 만듭니다
test_sentence.split(" ")


# In[19]:


# 자기 소개 문자열 self_data입니다
self_data = "My name is Yamada"

# self_data를 분리해 리스트를 작성하세요
word_list = self_data.split(" ")

# "이름"을 출력하세요
print(word_list[3])


# In[21]:


# re 모듈을 import합니다
import re
# 나눌 문자열입니다
test_sentence = "this,is a.test,sentence"
# ","와 " "와 "."으로 분할해 리스트를 만듭니다
re.split("[, .]", test_sentence)


# In[23]:


import re
# 시간 데이터가 저장된 문자열 time_data입니다
time_data = "2017/4/1_22:15"

# time_data를 분리하여 리스트를 작성하세요
time_list = re.split("[/_:]",time_data)

# "월"과 "시"를 출력하세요
print(time_list[1])
print(time_list[3])


# In[24]:


# for 루프로 함수를 적용합니다
a = [1, -2, 3, -4, 5]
new = []
for x in a:
    new.append(abs(x))
print(new)


# In[25]:


# map으로 함수를 적용합니다
a = [1, -2, 3, -4, 5]
list(map(abs, a))


# In[26]:


import re

# 배열 time_list
time_list = [
    "2006/11/26_2:40",
    "2009/1/16_23:35",
    "2014/5/4_14:26",
    "2017/8/9_7:5",
    "2017/4/1_22:15"
]

# 문자열에서 "시"를 추출하는 함수를 작성하세요
get_hour = lambda x: int(re.split("[/_:]",x)[3]) # int()로 string 형을 int 형으로 변경

# 위에서 만든 함수를 이용하여 각 요소에서 "시"만 꺼내 배열로 만드세요
hour_list = list(map(get_hour, time_list))

# 출력하세요
print(hour_list)


# In[27]:


# for 루프로 필터링합니다
a = [1, -2, 3, -4, 5]
new = []
for x in a:
    if x > 0:
        new.append(x)


# In[29]:


# filter 함수로 필터링합니다
a = [1, -2, 3, -4, 5]
print(list(filter(lambda x: x>0, a)))


# In[31]:


import re

# time_list..."년/월/일_시:분"
time_list = [
    "2006/11/26_2:40",
    "2009/1/16_23:35",
    "2014/5/4_14:26",
    "2017/8/9_7:5",
    "2017/4/1_22:15"
]
# 문자열의 "월"이 조건을 채울 때 True를 반환하는 함수를 작성하세요
is_first_half = lambda x: int(re.split("[/_:]", x)[1]) - 7 < 0

# 위에서 만든 함수로 조건에 맞는 요소를 찾아내, 배열로 만듭니다
first_half_list = list(filter(is_first_half, time_list))

# 출력하세요
print(first_half_list)


# In[32]:


# 중첩된 배열입니다
nest_list =[
    [0, 9],
    [1, 8],
    [2, 7],
    [3, 6],
    [4, 5]
]

# 두번째 요소를 키로 하여 정렬합니다
print(sorted(nest_list, key = lambda x: x [1]))


# In[34]:


# time_data...[년, 월, 일, 시, 분]
time_data = [
    [2006, 11, 26, 2, 40],
    [2009, 1, 16, 23, 35],
    [2014, 5, 4, 14, 26],
    [2017, 8, 9, 7, 5],
    [2017, 4, 1, 22, 15]
]

# "시"를 키로 정렬하고 배열로 만듭니다
sort_by_time = sorted(time_data, key=lambda x: x[3])

# 출력하세요
print(sort_by_time)


# In[35]:


# 리스트 내포로 각 요소의 절대값을 취합니다
a = [1, -2, 3, -4, 5]
print([abs(x) for x in a])


# In[36]:


# map으로 리스트를 만듭니다
a = [1, -2, 3, -4, 5]
print(list(map(abs, a)))


# In[37]:


# minute_data, 단위는 분입니다
minute_data = [30, 155, 180, 74, 11, 60, 82]

# 분을 [시, 분]으로 변환하는 함수를 작성하세요
h_m_split = lambda x: [x // 60, x % 60]

# 리스트 내포를 사용하여 배열을 작성하세요
h_m_data = [h_m_split(x) for x in minute_data]

# 출력하세요
print(h_m_data)


# In[38]:


# 리스트 내포 필터링(후위 if)
a = [1, -2, 3, -4, 5]
print([x for x in a if x > 0])


# In[41]:


# minute_data, 단위는 분입니다
minute_data = [30, 155, 180, 74, 11, 60, 82]

# 리스트 내포를 사용하여 배열을 작성하세요
just_hour_data = [x for x in minute_data if x % 60 == 0 ]

# 출력하세요
print(just_hour_data)


# In[42]:


# zip을 이용한 동시 루프입니다
a = [1, -2, 3, -4, 5]
b = [9, 8, -7, -6, -5]
for x, y in zip(a, b):
    print(x, y)


# In[43]:


# 리스트 내포로 동시에 처리합니다
a = [1, -2, 3, -4, 5]
b = [9, 8, -7, -6, -5]
print([x**2 + y**2 for x, y in zip(a, b)])


# In[44]:


# 시간 데이터 hour, 분 데이터 minute
hour = [0, 2, 3, 1, 0, 1, 1]
minute = [30, 35, 0, 14, 11, 0, 22]

# 시, 분을 인수로 받은 뒤, 분으로 변환하는 함수를 작성하세요
h_m_combine = lambda x, y: x*60 + y

# 리스트 내포를 사용하여 배열을 작성하세요
minute_data1 = [h_m_combine(x, y) for x, y in zip(hour, minute)]

# 출력하세요
print(minute_data1)


# In[46]:


a = [1, -2, 3]
b = [9, 8]

# 이중 루프입니다
for x in a:
    for y in b:
        print(x, y)


# In[47]:


# 리스트 내포로 이중 루프를 만듭니다
print([[x, y] for x in a for y in b])


# In[49]:


# 이진수의 자리입니다
fours_place = [0, 1]
twos_place = [0, 1]
ones_place = [0, 1]

# 리스트 내포에서 다중 루프를 사용하여 0에서 7까지의 정수를 계산하여 배열로 만듭니다
digit = [x*4 + y*2 + z for x in fours_place for y in twos_place for z in ones_place]

# 출력하세요
print(digit)


# In[50]:


# 딕셔너리의 요소가 출현한 횟수를 기록합니다
d = {}
lst = ["foo", "bar", "pop", "pop", "foo", "popo"]
for key in lst:
    # d에 key가 존재하느냐에 따라 분류합니다
    if key in d:
        d[key] += 1
    else:
        d[key] = 1

print(d)


# In[52]:


from collections import defaultdict

# 딕셔너리의 요소가 출현한 횟수를 기록합니다
d = defaultdict(int)
lst = ["foo", "bar", "pop", "pop", "foo", "popo"]
for key in lst:
    d[key] += 1

print(d)


# In[54]:


from collections import defaultdict

# 문자열 description
description = "Artificial intelligence (AI, also machine intelligence, MI) is " + "intelligence exhibited by machines, rather than " + "humans or other animals (natural intelligence, NI)."

# defaultdict를 정의하세요
char_freq = defaultdict(int)

# 문자의 출현 횟수를 기록 하세요
for i in description:
    char_freq[i] += 1

# 정렬하고 상위 10개의 요소를 출력하세요
print(sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:10])


# In[55]:


from collections import defaultdict
defaultdict(list)


# In[56]:


# 딕셔너리에 value 요소를 추가합니다
d = {}
price = [
    ("apple", 50),
    ("banana", 120),
    ("grape", 500),
    ("apple", 70),
    ("lemon", 150),
    ("grape", 1000)
]

for key, value in price:
    # key의 존재 유무로 분기합니다
    if key in d:
        d[key].append(value)
    else:
        d[key] = [value]
        
print(d)


# In[58]:


from collections import defaultdict

# 요약할 데이터 price ...(이름, 값)입니다
price = [
    ("apple", 50),
    ("banana", 120),
    ("grape", 500),
    ("apple", 70),
    ("lemon", 150),
    ("grape", 1000)
]

# defaultdict를 정의하세요
d = defaultdict(list)

# 리스트 13.52처럼 value를 추가하세요
for key, value in price:
    d[key].append(value)

# 각 value의 평균을 계산하고, 배열로 만들어 출력하세요
print([sum(x) / len(x) for x in d.values()])


# In[4]:


# Counter를 import합니다
from collections import Counter

# 딕셔너리 요소의 출현 횟수를 기록합니다
lst = ["foo", "bar", "pop", "pop", "foo", "popo"]
d = Counter(lst)

print(d)


# In[8]:


# Counter로 문자열을 저장하고, 문자의 출현 빈도를 열거합니다
d = Counter("A Counter is a dict subclass for counting hashable objects.")

# 가장 많은 출현 빈도를 가진 요소 5개를 나열합니다
print(d.most_common(5))


# In[10]:


from collections import Counter
# 문자열 description입니다
description = "Artificial intelligence (AI, also machine intelligence, MI) is " + "intelligence exhibited by machines, rather than " + "humans or other animals (natural intelligence, NI)." 

# Counter를 정의하세요
char_freq = Counter(description)

# 정렬하고 출현 빈도 상위 10개를 출력하세요
print(char_freq.most_common(10))


# In[11]:


# if와 lambda를 사용하여 계산하세요(인수 a가 8 미만이면 5를 곱하고, 8 이상이면 2로 나누기)
a = 8
basic = lambda x: x * 5 if x < 8 else x / 2
print('계산결과')
print(basic(a))

import re
# 배열 time_list
time_list = [
    "2018/1/23_19:40",
    "2016/5/7_5:25",
    "2018/8/21_10:50",
    "2017/8/9_7:5",
    "2015/4/1_22:15"
]

# 문자열에서 "월"을 꺼내는 함수를 작성하세요
get_month = lambda x: int(re.split("[/_:]",x)[1])

# 각 요소의 "월"을 꺼내 배열로 만듭니다
month_list = list(map(get_month, time_list))

# 출력하세요
print()
print('월')
print(month_list)

# 리스트 내포를 사용하여 부피를 계산하세요
length= [3, 1, 6, 2, 8, 2, 9]
side = [4, 1, 15, 18, 7, 2, 19]
height = [10, 15, 17, 13, 11, 19, 18]

# 부피를 계산하세요
volume = [x * y * z for x, y,z in zip(length, side, height)]

# 출력하세요
print()
print('부피')
print(volume)

# 각 value의 평균값을 계산하고 price 리스트 내의 과일 이름을 열거하세요
from collections import defaultdict
from collections import Counter

# 요약할 데이터 price
price = [
    ("strawberry", 520),
    ("pear", 200),
    ("peach", 400),
    ("apple", 170),
    ("lemon", 150),
    ("grape", 1000),
    ("strawberry", 750),
    ("pear", 400),
    ("peach", 500),
    ("strawberry", 70),
    ("lemon", 300),
    ("strawberry", 700)
]

# defaultdict를 정의하세요
d = defaultdict(list)

# value에 가격을, key에 과일 이름을 추가하세요
price_key_count = []
for key, value in price:
    d[key].append(value)
    price_key_count.append(key)

# 각 value의 평균을 계산하고, 배열로 만들어 출력하세요
print()
print('value의 평균값')
print([sum(x) / len(x) for x in d.values()])

# 위의 price 리스트 중에서 과일 이름을 열거합니다
key_count = Counter(price_key_count)
print()
print('과일 이름')
print(key_count)


# In[ ]:




