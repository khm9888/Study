
import random as r

x =r.randint(1,100) #정수 1~100 사이의 값이 들어간다.
y =r.randint(1,100) #정수 1~100 사이의 값이 들어간다.

if x<y:
    temp = x
    x=y
    y=temp

ans=x-y

z=int(input(str(x)+"-"+str(y)"의 값은?"))
z=int(input("%d-%d의 값은?"%(x,y))

if ans==z :
    print("ok")
else:
    print("Bbbbbbbb")  

    
