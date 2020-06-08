
'''
first=0 #1등 점수
last=0 #5등 점수

for i in range(5):
    x = int(input("학생의 점수는?"))
    if first==0:
        first=x #처음에 값이 없을 때는 값을 넣어준다.
    else:
        if first<x: #새로운 값이 기존의 1등의 점수보다 높다면 새로운 값을 1등 변수(first)에 대입한다.
            first = x  
    if last==0:#처음에 값이 없을 때는 값을 넣어준다.
        last=x
    else:
        if last >x:
            last=x

print("1등의 점수는 %d 5등의 점수는 %d 입니다." %(first, last))
'''

dic = {}
dic['key']='item1','item2','item3'
print(dic['key'])
print(dic['key'][0])
