start = input("시작값")
end = input("끝나는값")
a=0
for i in range(start, end+1, 1) :
    if i%2==0 :
        a+=i
        print(a)
