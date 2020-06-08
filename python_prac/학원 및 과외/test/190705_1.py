

'''list =[]

for x in range(10):
    a=int(input("값을 입력해주세여"))
    list.append(a)
'''
'''
x = input("숫자를 입력하세요  :")
xx =int(x)
sum = 0
print(len(x))
print(type(len(x)))
for i in range(len(x)):
    if xx//10**int(len(x)-1)!=0:
        y=xx//10**int(len(x)-1)
        print("y값은 %d입니다."%y)
        xx=xx%10**int(len(x)-1)
        print("xx값은 %d입니다."%xx)
        sum += y
    else:
        sum += xx

print("sum은"+str(sum))

'''
'''
x = input("숫자를 입력하세요  :")
sum = 0
for i in x:
    sum +=int(i)
print(sum)
'''
'''
import turtle as t


def squ (x):
    for i in range(4):
        t.fd(x)
        t.lt(90)

def draw (x,y):
    t.up()
    t.goto(x,y)
    t.down()
    t.begin_fill()
    t.color("green")
    squ(50)
    t.end_fill()

s = t.Screen()
s.onscreenclick(draw)
'''
import random as r
nlist=[]
i=0
while len(nlist)<6:
    x = r.randint(1,6)
    nlist.append(x)
    #print(nlist)
    for j in range(i+1):
        if j < i  and nlist[j]==nlist[i]:
            nlist.pop()
            i-=1
    i+=1
print(nlist)
