# 두 숫자를 입력(input) 받는다. 그리고 덧셈/뺄셈(+/-)을 할지 
# 선택을 받고, 선택 받은 기호(덧셈/뺄셈)에 따라 두 수의 합 또는 차를 구하라

a = int(input())
b = int(input())
choice = input("덧셈 또는 뺄셈을 선택하시오 (+/-)")
if choice =="덧셈"or choice == "+" :
    print("두 수의 합은"+str(a+b))
elif choice =="뺄셈" or choice == "-" : 
    print("두 수의 차는"+str(a-b))
else : 
    print("잘못입력하셨습니다.")
