#3
import random
import string

upper=string.ascii_uppercase
one = random.choice(upper)

num = input("Type a number:")

n=ord(one)
if n==90:
    m=65
else:
    m=n+1
n_str = chr(n)
m_str = chr(m)
two=m_str
three = str(n+m)[0] 
four = random.choice(['&', '*', '@', '$','#'])
five = random.choice(str(num))

print(type(one))
print(type(two))
print(type(three))
print(type(four))
print(type(five))

print("The random token is:"+one+two+three+four+five)
    