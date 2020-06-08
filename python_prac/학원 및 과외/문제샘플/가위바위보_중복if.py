#컴퓨터와 가위바위보 하여 이길 경우, 비길 경우, 질 경우에 대한 결과를
#출력하시오.

import random as r

c = r.randint(1,3)

u = input("가위~바위~보???(r/s/p)")
u1=0

if u =="r":
    u1=1
elif u=="s":
    u1=2
elif u=="p":
    ul=3
else:
    print("잘못냈어요")
if u1 !=0:
    if c==u1:
        print("무승부입니다")
    else:
        if (c==1 and u1==2) or (c==2 and u1==3) or (c==3 and u1==1):
            print("컴퓨터 승리입니다")
        else:
            print("유저 승리입니다")
        
