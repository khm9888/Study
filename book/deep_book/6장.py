#!/usr/bin/env python
# coding: utf-8

# In[1]:


vege = "potato"
n = [4, 5, 2, 7, 6]

# 변수 vege의 오브젝트의 길이를 출력하세요
print(len(vege))

# 변수 n의 오브젝트의 길이를 출력하세요
print(len(n))


# In[2]:


# append의 복습
alphabet = ["a","b","c","d","e"]
alphabet.append("f")
print(alphabet)


# In[4]:


# sorted 입니다
number = [1,5,3,4,2]
print(sorted(number))
print(number)


# In[5]:


# sort 입니다
number = [1,5,3,4,2]
number.sort()
print(number)


# In[1]:


# 메서드의 예입니다
city = "Tokyo"
print(city.upper())
print(city.count("o"))


# In[3]:


animal = "elephant"

# 변수 animal_big에는 변수 animal의 문자열을 대문자로 만들어 저장하세요
animal_big = animal.upper() 
print(animal)
print(animal_big)

# 변수 animal에 'e'가 몇 개 포함되어 있는지 출력하세요
print(animal.count("e"))


# In[1]:


print("{}에서 태어나 {}에서 유년기를 보냈습니다".format("서울", "광명시"))


# In[2]:


fruit = "banana"
color = "yellow"

# "banana는 yellow입니다"라고 출력하세요
print("{}는 {}입니다".format(fruit, color))


# In[3]:


alphabet = ["a", "b", "c", "d", "d"]
print(alphabet.index("a"))
print(alphabet.count("d"))


# In[4]:


n = [3, 6, 8, 6, 3, 2, 4, 6]

# "2"의 인덱스 번호를 출력하세요
print(n.index(2))

# 변수 n에 "6"이 몇 개 있는지 출력하세요
print(n.count(6))


# In[5]:


# sort()의 예
list = [1, 10, 2, 20]
list.sort()
print(list)


# In[7]:


# reverse()의 예
list = ["가", "나", "다", "라", "마"]
list.reverse()
print(list)


# In[8]:


n = [53, 26, 37, 69, 24, 2]

# n을 정렬하여 오름차순으로 출력하세요
n.sort()
print(n)

# n의 순서를 반대로 하여, 내림차순으로 출력하세요
n.reverse()
print(n)


# In[10]:


def sing():
    print("노래합니다!")
    
sing()


# In[13]:


# "홍길동입니다"라고 출력하는 함수 introduce를 작성하세요
def introduce():
    print("홍길동입니다")

introduce()


# In[12]:


def introduce(n):
    print(n + "입니다")

introduce("홍길동")


# In[14]:


# 인수 n을 세제곱한 값을 출력하는 함수 cube_cal을 만드세요
def cube_cal(n):
    print(n ** 3) 

# 함수를 호출합니다
cube_cal(4)


# In[1]:


def introduce(first, second):
    print("성은 " + first + "이고, 이름은 "+ second + "입니다.")
    
introduce("홍", "길동")


# In[19]:


# 함수 introduce를 만드세요
def introduce(n, age):
    print(n + "입니다. "+ str (age) + "살입니다.")

# 함수를 호출하세요
introduce("홍길동", 18)


# In[22]:


def introduce(first = "김", second = "길동"):
    print("성은 " + first + "이고, 이름은 " + second + "입니다.")

introduce("홍")


# In[23]:


def introduce(first, second = "길동"):
    print("성은 " + first + "이고, 이름은 " + second + "입니다.")

introduce("홍")


# In[24]:


def introduce(first = "홍", second):
    print("성은 " + first + "이고, 이름은 " + second + "입니다.")


# In[31]:


# 초기값을 설정하세요
def introduce(age, n):
    print(n + "입니다." + str(age) + "살입니다.")

# 함수를 호출합니다


# In[26]:


# 초기값을 설정하세요
def introduce(age, n = "홍길동"):
    print(n + "입니다. " + str(age) + "살입니다.")

# 함수를 호출합니다
introduce(18)


# In[28]:


def introduce(first = "김", second = "길동"):
    return "성은 " + first + "이고, 이름은 " + second + "입니다."

print(introduce("홍"))


# In[29]:


def introduce(first = "김", second = "길동"):
    comment = "성은 " + first + "이고, 이름은 " + second + "입니다."
    return comment

print(introduce("홍"))


# In[30]:


# bmi을 계산하는 함수를 만들고, bmi를 반환 값으로 만드세요
def bmi(height, weight):
    return weight / height**2 

print(bmi(1.65, 65))


# In[31]:


# time 패키지를 import합니다
import time

# time() 모듈을 사용하여 현재 시간을 now_time에 대입합니다
now_time = time.time()

# print()를 이용하여 출력하세요
print(now_time)


# In[32]:


# from을 사용하여 time 모듈을 import합니다
from time import time

# from에서 import했기 때문에 패키지명을 생략할 수 있습니다
now_time = time()

print(now_time)


# In[33]:


# from을 이용하여 time 모듈을 import하세요
from time import time

# now_time에 현재 시간을 대입하세요
now_time = time()

print(now_time)


# In[34]:


# 값을 저장할 수 있습니다
mylist = [1, 10, 2, 20]

# 저장한 값에 정렬을 수행할 수 있습니다
mylist.sort()

# 함수로 전달하여 처리 결과를 표시할 수 있습니다
print(mylist)


# In[12]:


# MyProduct 클래스를 정의합니다
class MyProduct:
    # 생성자를 수정하세요
    def __init__(self, name, price, stock):
        # 인수를 멤버에 저장하세요
        self.name = name
        self.price = price
        self.stock = stock
        self.sales = 0

# MyProduct를 호출하여 객체 product_1을 만듭니다
product_1 = MyProduct("cake", 500, 20) 

# product_1의 stock을 출력하세요
print(product_1.stock)


# In[13]:


# MyProduct 클래스를 정의합니다
class MyProduct:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
        self.sales = 0
    # 구매 방법
    def buy_up(self, n):
        self.stock += n
    # 매각 메서드
    def sell(self, n):
        self.stock -= n
        self.sales += n*self.price
    # 요약 메서드
    def summary(self):
        message = "called summary().\n name: " + self.name +         "\n price: " + str(self.price) +         "\n stock: " + str(self.stock) +         "\n sales: " + str(self.sales)
        print(message)


# In[21]:


# MyProduct 클래스를 정의합니다
class MyProduct:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
        self.sales = 0
    # 요약 메서드
    # 문자열과 "자신의 방법"과 "자신의 구성원"을 연결하여 출력하세요
    def summary(self):
        message = "called summary()." +         "\n name: " + self.get_name() +         "\n price: " + str(self.price) +         "\n stock: " + str(self.stock) +         "\n sales: " + str(self.sales)
        print(message)
    # name을 반환하는 get_name()를 작성하세요
    def get_name(self):
        return self.name

    # 인수만큼 price를 뺴주는 discount()를 작성하세요
    def discount(self, n):
        self.price -= n

product_2 = MyProduct("phone", 30000, 100)
# 5,000 만큼 discount하세요
product_2.discount(5000)
# product_2의 summary를 출력하세요
product_2.summary()


# In[23]:


# MyProduct 클래스를 상속하는 MyProductSalesTax을 정의합니다
class MyProductSalesTax(MyProduct):
    
    # MyProductSalesTax는 생성자의 네 번째 인수가 소비세 비율을 받습니다
    def __init__(self, name, price, stock, tax_rate):
        # super()를 사용하면 부모 클래스의 메서드를 호출할 수 있습니다
        # 여기서는 MyProduct 클래스의 생성자를 호출합니다
        super().__init__(name, price, stock)
        self.tax_rate = tax_rate

    # MyProductSalesTax에서 MyProduct의 get_name을 재정의(오버라이드)합니다
    def get_name(self):
        return self.name + "(세금 포함)"

    # MyProductSalesTax에서 get_price_with_tax를 새로 구현합니다
    def get_price_with_tax(self):
        return int(self.price * (1 + self.tax_rate))


# In[24]:


product_3 = MyProductSalesTax("phone", 30000, 100, 0.1)
print(product_3.get_name())
print(product_3.get_price_with_tax())
# MyProductSalesTax 클래스에서는 summary() 메서드는 정의되어 있지 않지만,
# MyProduct을 계승하고 있기 때문에 MyProduct의 summary() 메서드를 호출할 수 있습니다
product_3.summary()


# In[29]:


class MyProduct:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
        self.sales = 0

    def summary(self):
        message = "called summary().\n name: " + self.get_name() +         "\n price: " + str(self.price) +         "\n stock: " + str(self.stock) +         "\n sales: " + str(self.sales)
        print(message)

    def get_name(self):
        return self.name

    def discount(self, n):
        self.price -= n

class MyProductSalesTax(MyProduct):
    # MyProductSalesTax는 네 번째 인수가 소비 세율을 받게 합니다
    def __init__(self, name, price, stock, tax_rate):
        # super()를 사용하면 부모 클래스의 메서드를 호출할 수 있습니다
        # 여기에서는 MyProduct 클래스의 생성자를 호출합니다
        super().__init__(name, price, stock)
        self.tax_rate = tax_rate

    # MyProductSalesTax에서 MyProduct의 get_name을 재정의(오버라이드)합니다
    def get_name(self):
        return self.name + "(세금 포함)"

    # MyProductSalesTax에 get_price_with_tax를 새로 구현합니다
    def get_price_with_tax(self):
        return int(self.price * (1 + self.tax_rate))

    # MyProduct의 summary() 메서드를 재정의하고 summary가 세금 포함 가격을 출력하도록 만드세요
    def summary(self):
        message = "called summary().\n name: " + self.get_name() +         "\n price: " + str(self.get_price_with_tax()+0) +         "\n stock: " + str(self.stock) +         "\n sales: " + str(self.sales)
        print(message) 

product_3 = MyProductSalesTax("phone", 30000, 100, 0.1)
print(product_3.get_name())
print(product_3.get_price_with_tax())
product_3.summary()


# In[30]:


pai = 3.141592
print("원주율은 %f" % pai)
print("원주율은 %.2f" % pai)


# In[2]:


def bmi(height, weight):
    return weight / height**2

# "bmi는 **입니다"라고 출력시켜주세요
print("bmi는 %.4f입니다" % bmi(1.65, 65))


# In[3]:


# 함수 check_character를 작성하세요
def check_character(object, character):
    return object.count(character)

# 함수 check_character에 입력하세요
print(check_character([1, 3, 4, 5, 6, 4, 3, 2, 1, 3, 3, 4, 3], 3))
print(check_character("asdgaoirnoiafvnwoeo", "d"))


# In[4]:


# 함수 binary_search의 내용을 작성하세요
def binary_search(numbers, target_number):
    # 최소치를 임시로 결정해 둡니다
    low = 0
    # 범위 내의 최대치
    high = len(numbers)
    # 목적지를 찾을 때까지 루프
    while low <= high:
        # 중앙값을 구합니다(index)
        middle = (low + high) // 2
        # numbers(검색 대상)의 중앙값과 target_number(찾는 값)가 동일한 경우
        if numbers[middle] == target_number:
            # 출력합니다
            print("{1}은(는) {0}번째에 있습니다".format(middle, target_number))
            # 종료합니다
            break
        # numbers의 중앙값이 target_number보다 작은 경우
        elif numbers[middle] < target_number:
            low = middle + 1
        # numbers의 중앙값이 target_number보다 큰 경우
        else:
            high = middle - 1

# 검색 대상 데이터
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# 찾을 값
target_number = 11
# 바이너리 검색 실행
binary_search(numbers, target_number)


# In[ ]:




