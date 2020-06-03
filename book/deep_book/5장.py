#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 변수 c에 "red" "blue" "yellow" 세 개의 문자열을 저장하세요
c = ["red", "blue", "yellow"]

print(c)

# c의 자료형을 출력하세요
print(type(c))


# In[3]:


n = 3
print(["사과", n, "고릴라"])


# In[4]:


apple = 4
grape = 3
banana = 6

# 리스트 형 fruits 변수에 apple, grape, banana 변수를 순서대로 저장하세요

fruits = [apple, grape, banana]
print(fruits)


# In[1]:


print([1, 2, 3, 4, 5, 6])


# In[2]:


fruits_name_1 = "사과"
fruits_num_1 = 2
fruits_name_2 = "귤"
fruits_num_2 = 10

# [["사과", 2], ["귤", 10]]이 출력되도록 fruits에 변수를 리스트에 대입합니다
fruits = [[fruits_name_1, fruits_num_1], [fruits_name_2, fruits_num_2]] 

# 출력합니다
print(fruits)


# In[3]:


a = [1, 2, 3, 4]
print(a[1])
print(a[-2])


# In[4]:


fruits = ["apple", 2, "orange", 4, "grape", 3, "banana", 1]

# 변수 fruits의 두 번째 요소를 출력합니다
print(fruits[1]) # print(fruits[-7])도 정답입니다

# 변수 fruits의 마지막 요소를 출력하세요
print(fruits[7]) # print(fruits[-1])도 정답입니다


# In[5]:


alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
print(alphabet[1:5])
print(alphabet[1:-5])
print(alphabet[:5])
print(alphabet[6:])
print(alphabet[0:20])


# In[1]:


chaos = ["cat", "apple", 2, "orange", 4, "grape", 3, "banana", 1,
"elephant", "dog"]

# chaos 리스트에서 ["apple", 2, "orange", 4, "grape", 3, "banana", 1] 리스트를 꺼내 변수 fruits에 저장하세요
fruits = chaos[1:9] #chaos[1:-3]도 가능

# 변수 fruits을 출력
print(fruits)


# In[1]:


alphabet = ["a", "b", "c", "d", "e"]
alphabet[0] = "A"
alphabet[1:3] = ["B", "C"]
print(alphabet)

alphabet = alphabet + ["f"]
alphabet += ["g","h"]
alphabet.append("i")
print(alphabet)


# In[2]:


c = ["dog", "blue", "yellow"]

# 변수 c의 첫 번째 요소를 "red"로 바꾸세요
c[0] = "red"
print(c)

# 리스트의 끝에 문자열 "green"을 추가하세요
c = c + ["green"] # c.append("green") 도 가능
print(c)


# In[3]:


alphabet = ["a", "b", "c", "d", "e"]
del alphabet[3:]
del alphabet[0]
print(alphabet)


# In[1]:


c = ["dog", "blue", "yellow"]
print(c)

# 변수 c의 첫 번째 요소를 제거하세요
del c[0]
print(c)


# In[2]:


alphabet = ["a", "b", "c"]
alphabet_copy = alphabet
alphabet_copy[0] = "A"
print(alphabet)


# In[3]:


alphabet = ["a", "b", "c"]
alphabet_copy = alphabet[:]
alphabet_copy[0] = "A"
print(alphabet)


# In[4]:


c = ["red", "blue", "yellow"]

# 변수 c의 값이 변하지 않도록 수정하세요
c_copy = list(c) #c[:] 도 정답입니다

c_copy[1] = "green"
print(c)


# In[5]:


dic ={"Japan": "Tokyo", "Korea": "Seoul"}
print(dic)


# In[6]:


# 변수 town에 딕셔너리를 저장하세요
town = {"Aichi": "Nagoya", "Kanagawa": "Yokohama"} 

# town 출력
print(town)
# 형의 출력
print(type(town))


# In[7]:


dic ={"Japan": "Tokyo", "Korea": "Soul"}
print(dic["Japan"])


# In[9]:


town = {"Aichi": "Nagoya", "Kanagawa": "Yokohama"}

# "Aichi의 현청 소재지는 Nagoya입니다"라고 출력하세요
print("Aichi의 현청 소재지는 " + town["Aichi"] + "입니다")

# "Kanagawa의 현청 소재지는 Yokohama입니다"라고 출력하세요
print("Kanagawa의 현청 소재지는 " + town["Kanagawa"] + "입니다")


# In[10]:


dic ={"Japan":"Tokyo","Korea":"Soul"}
dic["Japan"] = "Osaka"
dic["China"] = "Beijin"
print(dic)


# In[11]:


town = {"Aichi": "aichi", "Kanagawa": "Yokohama"} 

# 키 "Hokkaido" 값 "Sapporo"를 추가하세요
town["Hokkaido"] = "Sapporo" 
print(town)

# 키 "Aichi"의 값을 "Nagoya"로 변경하세요
town["Aichi"] = "Nagoya" 
print(town)


# In[13]:


dic ={"Japan": "Tokyo", "Korea": "Seoul", "China": "Beijin"}
del dic["China"]
print(dic)


# In[1]:


town = { "Aichi": "aichi", "Kanagawa": "Yokohama", "Hokkaido": "Sapporo"}

# 키가 "Aichi"인 요소를 제거하세요
del town["Aichi"]
print(town)


# In[2]:


n = 2
while n > 0:
    print(n)
    n -= 1


# In[3]:


x = 5
while x > 0:
    print("Aidemy")
    x -= 2


# In[4]:


x =  5

# while을 사용하여 변수 x가 0이 아닐 동안 루프하도록 만들어주세요
while x != 0:
    # 반복문 속에서 변수 x에서 1을 빼고, 출력합니다
    x -= 1
    print(x)


# In[1]:


x = 5

# while을 사용하여 변수 x가 0이 아닐 동안 루프하도록 만들어주세요
while x != 0:
    # 반복문 속에서 변수 x에서 1을 빼고, 출력합니다
    x -= 1
    if x != 0:
        print(x)
    else:
        print("Bang")


# In[6]:


animals = ["tiger", "dog", "elephant"]
for animal in animals:
    print(animal)


# In[7]:


storages = [1, 2, 3, 4]

# for 문으로 변수 storages의 요소를 출력하세요
for n in storages:
    print(n)


# In[9]:


storages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for n in storages:
    print(n)
    if n >= 5:
        print("다음은 여기")
        break


# In[13]:


storages = [1, 2, 3, 4, 5, 6]

for n in storages:
    print(n)
    # 변수 n의 값이 4일 때 처리를 종료하세요
    if n == 4:
        break;


# In[15]:


storages = [1, 2, 3]
for n in storages:
    if n == 2:
        continue
    print(n)


# In[16]:


storages = [1, 2, 3, 4, 5, 6]

for n in storages:
    # 변수 n이 2의 배수일 때만 처리를 생략하세요
    if n % 2 == 0:
        continue
    print(n)


# In[17]:


list = ["a", "b"]
for index, value in enumerate(list):
    print(index, value)


# In[18]:


animals = ["tiger", "dog", "elephant"]

# enumerate() 함수를 이용하여 출력하세요
for index, animal in enumerate(animals):
    print("index:" + str(index), animal)


# In[19]:


list = [[1, 2, 3],
        [4, 5, 6]]
for a, b, c in list:
    print(a, b, c)


# In[20]:


fruits = [["strawberry", "red"],
          ["peach", "pink"],
          ["banana", "yellow"]]

# for 문을 사용하여 출력하세요
for fruit, color in fruits:
    print(fruit + " is " + color)


# In[1]:


fruits = {"strawberry": "red", "peach": "pink", "banana": "yellow"}
for fruit, color in fruits.items():
    print(fruit + " is " + color)


# In[2]:


town = {"Aichi": "Nagoya", "Kanagawa": "Yokohama", "Hokkaido":
"Sapporo"}

# for 문을 사용하여 출력하세요
for prefecture, capital in town.items():
    print(prefecture, capital)


# In[8]:


items = {"지우개" : [100, 2], "펜" : [200, 3], "노트" : [400,5]}

total_price = 0

# 변수 items를 for 문으로 루프시킵니다.
for item in items:

    # "**는 하나에 ** 원이며, **개 구입합니다"라고 출력하세요
    print(item + "은(는) 하나에 " + str(items[item][0]) + "원이며, "
    + str(items[item][1]) + "개 구입합니다")

    # 변수 total_price에 가격×수량을 더해 저장하세요
    total_price += items[item][0] * items[item][1] 

# "지불해야 할 금액은 **원입니다"라고 출력하세요
print("지불해야 할 금액은 " + str(total_price) + "원입니다")

# 변수 money에 임의의 값을 대입하세요.
money = 4000

# money > total_price 일 때 "거스름돈은 **원입니다"라고 출력하세요
if money > total_price: 
    print("거스름돈은 " + str(money - total_price) + "원입니다")

# money == total_price 일 때 "거스름돈은 없습니다"라고 출력하세요.
elif money == total_price:
    print("거스름돈은 없습니다")

# money < total_price 일 때 "돈이 충분하지 않습니다"라고 출력하세요 
else:
    print("돈이 충분하지 않습니다")


# In[6]:


items = {"eracer" : [100, 2], "pen" : [200, 3], "notebook" : [400, 5]}
print(items["pen"][1])


# In[ ]:




