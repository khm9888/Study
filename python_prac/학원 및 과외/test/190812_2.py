'''
x= int(input("몇 팩토리얼을 만들고 싶어요?"))
fact=1
for i in range(1,x+1):
	fact *= i #fact = fact * i
	
	x=4
	fact=1
	i=1
	fact=1
	i=2
	fact=2
	i=3
	fact=6
	i=4
	fact=24
	
print("%d!은 %d입니다." %(x,fact))
'''
'''
sample = "아니"
while sample =="아니":
	sample = input("배고파?") #"응", "어" 아니.
print("밥먹자")
'''
pw="pythonisfun"

password=input("비밀선호를 입력하세요")
while password!=pw: 
    print("잘못입력하셨습니다.")
    password=input("비밀번호를 다시 넣으시오")
print("패스워드가 맞습니다")
