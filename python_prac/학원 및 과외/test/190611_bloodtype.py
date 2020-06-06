
cnt=0
while cnt < 4 :
    name=input("성함은요?")
    b_type = input("혈액형이 어떻게 되시나요?(A/AB/B/O형)")
    print("성함은 %s 입니다."%name)
    if (b_type =="A") | (b_type =="a") | (b_type =="A형") | (b_type =="a형") :
        print("혈액형은 %s 입니다."%b_type)
    elif (b_type =="B")| (b_type =="b") | (b_type =="B형") | (b_type =="b형") :
        print("혈액형은 %s 입니다."%b_type)
    elif b_type =="AB" or b_type =="ab" or b_type =="AB형" or b_type =="ab형" :
        print("혈액형은 %s 입니다."%b_type)
    elif b_type =="O" or b_type =="o" or b_type =="O형" or b_type =="o형" :
        print("혈액형은 %s 입니다."%b_type)
    else :
        print("잘못입력하신 거 같습니다. 다시 입력해주세요.")
        cnt -= 1
    cnt+=1
    print("cnt : %d"%cnt)
