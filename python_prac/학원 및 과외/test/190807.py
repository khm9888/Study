'''
for i in range(1,100):
        x=input(i)#사람이 입력받는 값
        y=(i%10)%3==0 and (i%10)!=0
        True False
        #print(type(y))
        if not y: #1,2,4,5,7,8,0
                continue
        elif y and x=="c": #3,6,9
                continue
        else:
                print("틀렸습니다.")
                break
		
for i in range(1,100):
        if str(i)[-1]="3" or str(i)[-1]="6" or str(i)[-1]="9":
'''
'''

ex>
1 (enter)
2 (enter)
3 c(enter)
4(enter)
5(enter)
6(enter)
실패하였습니다.

x=10
y=2

x,y,z = 10,20,30
'''
'''

컴퓨터가 1~100까지 숫자를 주는데, 3,6,9가 들어가는 숫자가 나오면
c라고 값을 입력 받아야 한다.
c라고 입력하지 못하면 실패로 간주하고 게임을 종료한다.
'''
#n_list = [0,0]
#두자리 숫자, 1의 자리, 10의 자리 숫자를 뜻함.

'''
3의 배수인 경우 // 박수를 치는 경우
'''
'''
for i in range(1,11):
        y=input(i)#1부터 숫자가 나옴
        x1=(i%10)%3==0  #1의 자리 숫자
        #x2=int(i/10)%3==0 #10의 자리 숫자
        #print(x1,x2)
        
        #if (x1 and x2) or c!="c":
        if x1 and y!="c":
                print("틀렸습니다.")
                break
                '''

'''
1부터 100까지 숫자를 순차적으로 계산하여 합을 구한다.
단, 17이 되면 멈춘다. #break
''''''
s=0
for i in range(1,100+1):
        if i==17:
                break
        s=s+i
        print("i의 값은 %d이고, sum의 값은 %d입니다." %(i,s))
''''''
1부터 100의 숫자중에 입력받은 배수의 숫자만 출력해라.
'''
a=int(input("배수를 적어라")) #4

for i in range(1,10):
        print("i의 값은 %d이고, a의 값은 %d입니다." %(i,a))
        if i%a==0:
                print("*********** "+str(i)+" ********")
        

try:
        print(su+1)

catch:

        print("에러가 났습니다")
