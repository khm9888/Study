'''5명의 학생의 점수를 입력받는다. 그리고 1등의 점수와
5등의 점수를 출력하라(for문/중복if문 사용)'''

'''
10/20/15/16/9
10/10
20/10
20/9
'''

first=0
last=0


for i in range(5):
    x = int(input("학생의 점수는?"))
    if first==0:
        first=x
    else:
        if first<x:
            first = x
            
    if last==0:
        last=x
    else:
        if last >x:
            last=x


    
