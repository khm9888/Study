#로그인 
#아이디는 "tree", 비밀번호는 3434 로 설정해주고, 로그인 시도 프로그램을 만들어라.
#아이디, 비밀번호의 확인은 if 문을 사용해서 시도한다.  값의 입력은 input() 통해서 받는다.

i="tree"
pw=3434

i_u=input("아이디를 입력해주세요: ")
pw_u=input("비밀번호를 입력해주세요: ")
#elif
if i_u==i and pw==pw_u:
    print("로그인 성공하셨습니다.")
elif i_u != i:
    print("아이디를 틀렸습니다.")
elif pw_u != i:
    print("패스워드를 틀렸습니다.")
else:
    print("잘못된 경우입니다.")

#중복 if
if i_u==i:
    if pw_u==pw:
        print("로그인 성공하셨습니다.")
    else:
        print("비밀번호 오류")
else:
    print("아이디 오류")
