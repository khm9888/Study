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
# from collections import Counter

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

#p37

def generate_range(n):
    i=0
    while i < n:
        yield  i
        i += 1

print(type(generate_range(10)))        
for i in range(10):
    print(f"i : {i}")


print(generate_range(10))
even = (i for i in generate_range(20) if i % 2 ==0)

print(even)
print(type(even))