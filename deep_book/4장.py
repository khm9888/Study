#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 파이썬에서는 행의 시작 부분에 #를 입력하면 주석으로 처리됩니다
# "Hello world"를 출력하세요
print("Hello world")


# In[2]:


# 5 + 2의 결과를 출력하세요
print(5 + 2)

# 3 + 8의 결과를 출력합니다
print(3 + 8)


# In[3]:


print(3 + 6)


# In[4]:


print("8 - 3")


# In[5]:


# 숫자 18을 출력하세요
print(18)

# 2 + 6을 계산하고, 결과를 출력하세요
print(2 + 6)

# “2 + 6”이라는 문자열을 출력하세요 
print("2 + 6")


# In[6]:


# 3 + 5
print(3 + 5)

# 3 - 5
print(3 - 5)

# 3 × 5
print(3 * 5)

# 3 ÷ 5
print(3 / 5)

# 3을 5로 나눈 나머지
print(3 % 5)

# 3의 5승
print(3 ** 5)


# In[7]:


n = "강아지"
print(n)


# In[8]:


# print를 변수명에 사용하면 print()를 호출하는 단계에서 오류가 발생합니다
#print = "Hello"
#print(print) # TypeError: 'str' object is not callable


# In[9]:


# 변수 n에 "고양이"를 대입하세요
n = "고양이"

# 변수 n을 출력하세요
print(n)

# "n"이라는 문자열을 출력하세요
print("n")

# 변수 n에 3 + 7이라는 수식을 대입하세요
n = 3 + 7

# 변수 n을 출력하세요
print(n)


# In[10]:


x = 1
print(x)
x = x + 1
print(x)


# In[11]:


m = "고양이"
print(m)

# 변수 m에 "강아지"을 덮어써서 출력하세요
m = "강아지"
print(m)

n = 14
print(n)

# 변수 n에 5를 곱하여 덮어하쓰세요. n = n * 5 또는 n = 5 * n도 정답입니다
n *= 5

print(n)


# In[12]:


m = "홍길동"
print("내 이름은 "+ m + "입니다")


# In[14]:


# 변수 p에 “서울”을 대입하세요
p = "서울"

# 변수 p를 사용하여 "저는 서울 출신입니다"라고 출력하세요
print("저는 "+ p + " 출신입니다")


# In[16]:


height = 177
print("키는 "+ height + "cm입니다.") # "TypeError: must be str, not int"라는 오류 메시지가 출력됩니다


# In[18]:


height = 177
type(height) # int 형임을 알 수 있습니다


# In[3]:


h = 1.7
w = 60

# 변수 h, w의 형을 출력하세요
print(type(h))
print(type(w))

# 변수 bmi 계산 결과를 대입하세요
bmi = w / h ** 2

# 변수 bmi를 출력하세요
print(bmi)

# 변수 bmi의 형을 출력하세요
print(type(bmi))


# In[21]:


h = 177
print("신장은 " + str(h) + "cm입니다")


# In[22]:


a = 35.4
b = 10
print(a + b)


# In[25]:


h = 1.7
w = 60
bmi = w / h ** 2

# "당신의 bmi는 〇〇입니다"라고 출력하세요
print("당신의 bmi는 " + str(bmi) + "입니다")


# In[1]:


greeting = "yo!"
print(greeting * 2)


# In[2]:


n = "10"
print(n * 3)


# In[4]:


print(1 + 1 == 3)


# In[6]:


# "!=" 을 이용해 4+6과 -10의 관계식을 만들어 True를 출력하세요
print(4 + 6 != -10)


# In[7]:


n = 2
if n == 2: 
    print("아쉽습니다! 당신은 " + str(n) + "번째로 도착합니다") # n이 2일 때만 표시됩니다


# In[8]:


animal = "cat"
if animal == "cat":
    print("고양이는 귀엽습니다") # animal이 cat일 때만 나타납니다


# In[9]:


n = 16

# if를 사용하여 변수 n이 15보다 클 때 "큰 숫자"를 출력하세요
if n > 15 :
    print("큰 숫자")


# In[11]:


n = 2
if n == 1:
    print("우승을 축하합니다!") # n이 1인 경우에만 표시됩니다
else:
    print("아쉽습니다! 당신은 "+ str(n) + "번째로 도착했습니다") # n이 1이 아닐 때 나타납니다


# In[12]:


animal = "cat"
if animal == "cat":
    print("고양이는 귀엽습니다") # animal가 "cat"일 때만 표시됩니다
else:
    print("고양이가 아니다냥") # animal가 "cat"이 아닐 때 나타납니다


# In[2]:


n = 14

if n > 15:
    print("큰 숫자")

# else를 이용하여 "작은 숫자"를 출력하세요
else:
    print("작은 숫자")


# In[4]:


number = 2

if number == 1:
    print("금메달입니다!")
elif number == 2:
    print("은메달입니다!")
elif number == 3:
    print("동메달입니다!")
else:
    print("아쉽습니다! 당신은 "+ str (number) + "번째로 도착했습니다")


# In[5]:


animal = "cat"

if animal == "cat":
    print("고양이는 귀엽네요")
elif animal == "dog":
    print("개는 멋지네용")
elif animal == "elephant":
    print("코끼리는 크다구")
else:
    print("고양이도, 개도, 코끼리도 아니다냥")


# In[8]:


n = 14

if n > 15: 
    print("큰 숫자")

# elif를 사용하여 n이 11 이상 15 이하일 때 "중간 숫자"라고 출력하세요
elif n >= 11:
    print("중간 숫자")

else:
    print("작은 숫자")


# In[9]:


n_1 = 14
n_2 = 28 

# n_1가 8보다 크고 14보다 작다는 조건식을 만들고, and를 사용하여 출력하세요
print(n_1 > 8 and n_1 < 14)

# n_1의 제곱이 n_2의 다섯 배 보다 작다는 조건식을 만들고, not을 사용하여 출력하세요
print(not n_1 ** 2 < n_2 * 5)


# In[12]:


# 변수 year에 연도를 입력하세요
year = 2000

# if 문으로 조건 분기를 하고 윤년과 평년을 판별하세요 
if year % 100 == 0 and year % 400 != 0: 
    print(str(year) + "년은 평년입니다")
elif year % 4 == 0: 
    print(str(year) + "년은 윤년입니다")
else: 
    print(str(year) + "년은 평년입니다")


# In[ ]:




