'''
#3,6,9가 될 때 clap 진다!

c=""
for i in range(1,10):
    c==input(i)#1,2,3,4,5~9
    if c !="clap" and i%3==0:
        print("틀렸습니다.")
        break;
        '''

#숫자맞추기, 게임 (1,100), 
'''
컴퓨터로부터 랜덤값을 받고, 그 값을 맞추는 게임.
컴퓨터가 생각하는 값보다 낮으면 높음!이라고 출력
높으면! 낮음! 이라고 출력한다.
'''
'''
import random
x=random.randrange(1,101)#컴퓨터의 값
y=int(input("값을 맞춰주세요"))
while x!=y:
    if x>y:
        print("높음")
    else :
        print("낮음")
    y=int(input("값을 맞춰주세요"))
print("맞췄습니다.")
'''
name="홍유지"
age=10
print("내 나이는 "+age+"입니다.")


