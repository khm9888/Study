#2장

#15페이지

import this

'''
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
'''
'''
못생긴 것보다 아름답습니다.
암시적인 것보다 명시적인 것이 좋습니다.
단순보다 복잡합니다.
복잡한 것이 복잡한 것보다 낫습니다.
평평한 것이 중첩보다 낫습니다.
스파 스가 밀도보다 낫습니다.
가독성이 중요합니다.
특별한 경우는 규칙을 어길만큼 특별하지 않습니다.
실용성은 순도를 능가하지만.
오류가 자동으로 전달되지 않아야합니다.
명시 적으로 침묵하지 않는 한.
모호함에 직면하여 추측하려는 유혹을 거부하십시오.
그것을하는 명백한 방법이 있어야합니다.
네덜란드 인이 아니라면 처음에는 그 방법이 명확하지 않을 수 있습니다.
지금보다 결코 낫습니다.
지금은 * 오른쪽 *보다 나을 때가 많지 않습니다.
구현이 설명하기 어렵다면 나쁜 생각입니다.
구현이 설명하기 쉬운 경우 좋은 생각 일 수 있습니다.
네임 스페이스는 훌륭한 아이디어 중 하나입니다. 더 많은 것을 해보자!
'''

# #p18
# for i in range(1,6):
#     print(i)
#     for j in range(1,6):
#         print(j)
#         print(i+j)
#     print(i)
# print("done looping")

# #p19
# plus = 2 +\
#     3

# #p20

# import re as regex
# # my_regex = regex.compile("[0-9]+"regex.I)

# #p21

# y=lambda x:x+4
# print(y(4))

# # p.23

# try:
#     print(0/0)
# except ZeroDivisionError:
#     print("cannot divide by zero")

# #p24

# x=list(range(0,10))
# x[0]=-1
# print(x[::3])
# print(x)
# print(x[5:2:-1])

# x=[1,2,3]
# y=[4,5,6]
# x.extend(y)
# print(x)

# #p25

# def sum_and_product(x,y):
#     return x+y,x*y
# x,y=sum_and_product(2,3)
# print(type(x))

# #p26

# grades={"joel":80}

# print(grades.get("joel"))

# #p27

# x={"해민":3}
# print(x["해민"])

# words="sfdgsdgdsㅀㄹㅇㅇㄴㅁㅇ"
# word_counts={}
# for word in words:
#     if word in word_counts:
#         word_counts[word]+=1
#     else:
#         word_counts[word]=1
         
# print(word_counts)

# from collections import defaultdict

# word_counts2=defaultdict(int)
# for word in words:
#     if word in word_counts2:
#         word_counts[word]+=1
# print(word_counts2.items())

# p29

# 2.12.Counter
# from collections import Counter
# from collecitons import Counter

# c = Counter([1,2,3,4,1])

# print(c)

# words="sfdgsdgdsㅀㄹㅇㅇㄴㅁㅇ"

# c1=Counter(words)
# print(c1)

# print(c1.most_common(5))

# s=set()

# s.add(1)
# s.add(1)
# s.add(2)

# p30, 2.14 흐름제어
# parity=[0,1]
# parity = x if x%2 ==0 else "odd" 

# #p32

# w="test"

# print("te" or w)

# # safe_x = x if x is not None else 0

# #p33

# even= [x for x in range(5) if x%2==0]
# print(even)

# even_dic= {x for x in range(5) if x%2==0}
# print(even_dic)

# pairs = [(x,y) for x in range(2,11) for y in range(1,10)]

# print(len(pairs))

# #p34, assert

# assert 1+1 ==2
# assert 1+1 ==3, "1none"

# # p35 클래스

# class CounterClicker:
#     def __init__(self,count):
#         self.count=count
        
#     def __repr__(self):
#         return f"CounterClicker(count={self.count})"
    
#     def click(self, num_times =1):
#         self.count+=num_times
    
#     def read(self):
#         return self.count
    
#     def reset(self):
#         self.count=0

# c=CounterClicker(3)
# print(c.count)     
# assert c.read() ==0

# c.click()
# c.click()
# assert c.click().read==2


# class NoResetClicker(CountingClicker):
#     def reset(self):
#         pass

# #p37

# def generate_range(n):
#     i=0
#     while i < n:
#         yield  i
#         i += 1

# def generate_range(n):
#     i=0
#     while i<n:
#         yield i
#         i+=1


# print(type(generate_range(10)))        
# for i in range(10):
#     print(f"i : {i}")


# print(generate_range(10))
# even = (i for i in generate_range(20) if i % 2 ==0)

# print(even)
# print(type(even))

# names=[]
# names.append("Alice")
# names.append("Bob")
# names.append("Charlie")
# names.append("Debbie")

# for i,name in enumerate(names):
#     print(f"name {i} is {name}")

# #2.21 난수생성

# import random

# random.seed(10)

# print(random.sample(range(10),6))

# random.seed(10)

# print(random.sample(range(10),6))

#2.22 정규표현식

import re
re_examples = [
    not re.match("a","cat"),#"cat"은 "a"로 시작하지 않음
    re.search("a","cat"),
    3==len(re.split("[ab]","carbs"))#a 혹은 b로 분리하면 ['c','r','s']가 생성됨
    ,
    "R-D-"==re.sub("[0-9]","-","R2D2")
    ]

assert all(re_examples), "none"
print(re_examples)

list1=["a","b","c"]
list2=[1,2,3]

x=[i for i in zip(list1,list2)]

print(x)

x,y=zip(*x)

print(f"x:{x}")
print(f"y:{y}")