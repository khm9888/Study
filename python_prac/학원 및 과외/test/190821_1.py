
#3,6,9가 될 때 clap 진다!

c=""
for i in range(1,10):
    c==input(i)#1,2,3,4,5~9
    if c !="clap" and i%3==0:
        print("틀렸습니다.")
        break;
        
