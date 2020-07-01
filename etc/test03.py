N = int(input())
n_vote = []

for i in range(N):
    n_vote.append(int(input()))
    
cand = n_vote[1:len(n_vote)]
dasom = n_vote[0]
    
if(N == 1):
    print(0)

else:
    num = 0
    cand = sorted(cand, reverse = True)
    while(cand[0] >= dasom):
        dasom += 1
        cand[0] -= 1
        num += 1
        cand = sorted(cand, reverse = True)

    print(num)