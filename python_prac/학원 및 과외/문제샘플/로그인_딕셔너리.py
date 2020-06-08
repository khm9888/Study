#딕셔너리를 이용해서 아이디 비밀번호를 저장하고,
#각 아이디와 비빌번호를 입력받아 로그인 시도를 해보자


data = {"apple":(123,"남자"),"banana":(456,"여자")}
data = {"apple":"123","banana":"456"}

s=" "
while True:
    s=input("새로운 아이디?(그만하고 싶다면 stop)")
    if s=="stop":
        break;
    pw1=data[s]=input("새로운 비밀번호?")

print(data)

x=input("아이디는?")
y=int(input("비밀번호는?"))

if x in data.keys():
    if data[x]==y:
    #if data[x][0]==y:
        print("로그인 성공")
    else:
        print("비밀번호 틀렸습니다")
else:
    print("아이디가 존재하지 않습니다.")
