'''
for문으로 1부터 5까지 출력해보아라

for 문으로 1부터 5까지의 합을 구해보아라.


'''

for i in range(3,24,3): #3에서부터 21까지 중에서 3의 배수를 출력하라.
    print(i)

x=10

if True: #sa

print(x>5)

y= True
y2= False #boolean

i=0
while True:
    i=i+1
'''
'''


#sfdsgfdgfgfgfdgfgfg
x=input()
infile = open(x,"r")
for line in infile:
    line =line.rstrip()
    word_list=line.split()
    for word in word_list:
        print(word)
infile.close()
