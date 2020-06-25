persons=int(input())#몇명?
people = []#후보들 
for i in range(persons):
    people.append(int(input()))#숫자입력
    
dasom = people[0]
candidate = people[1:]
candidate = sorted(candidate)
cnt=0#몇번했는지 확인
if persons==1:
    pass
else:
    dasom = people[0]
    candidate = people[1:]
    candidate.sort()
    while dasom<=candidate[-1]:#무한루프
        candidate[-1]-=1
        dasom+=1
        candidate = sorted(candidate)
        cnt+=1      
        # print(people)
        # print(cnt,people)
print(cnt)
