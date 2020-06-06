#파일 입출력 테스트 p319


#infile = open("d://test.txt","r")#읽기
#infile = open("d://test.txt","w")#쓰기
#infile = open("d://test.txt","a")#추가
#infile = open("d://test.txt","r+")#읽기쓰기, seek()함수를 사용하면 변환가능.
'''
#전체를 읽어오는 함수
lines = infile.read()
print(lines)
'''

'''
# 여러줄을 입력받아서, 리스트 안에 저장
infile = open("d://test.txt","r")#읽기
lines = infile.readlines()
print(lines)
infile.close()
'''
#p321
'''
infile = open("d://test.txt","r")#읽기
line =infile.readline().rstrip()#오른쪽 공백 삭제 기호이지만, 여기서는 엔터\n을 삭제시켜준다.
while line !="":
    print(line)
    line=infile.readline().rstrip()

infile.close()
'''
'''
infile = open("d://test.txt","r")#읽기
#print(type(infile))
#x=list(infile)
#print(x[0])
for line in infile:
    line = line.rstrip()
    print(line)
infile.close()

'''
'''
infile = open("d://test.txt","r")#읽기
for line in infile:#파일 객체를 문자열의 모음(컨테이너)라고 처리하기 때문에 이처럼 처리가 가능하다.
    line = line.strip()
    print(line)
infile.close()
'''

'''
outfile = open("d://test2.txt","w")#쓰기
outfile.write("홍길동 010-2342-2324 \n")
outfile.write("박보검 021-3432-2324 \n")
outfile.write("서강준 013-6453-1234 \n")
outfile.close()

infile = open("d:/test2.txt")
print(infile.read())
infile.close()
'''
'''
outfile = open("d:/test.txt","a")

outfile.write("김철수 010-0853-1243 \n")
outfile.write("박영상 010-3434-1254 \n")

outfile.close()
'''
'''
outfile = open("d://proverbs.txt","w")#쓰기
outfile.write("All`s well that ends well.\n")
outfile.write("Bad news travels fast.\n")
outfile.write("Well begun is half done. \n")
outfile.write("Birds of a feather flock together. \n")

outfile.close()

infile = open("d://proverbs.txt","r")

for line in infile:
    line = line.strip()
    word_list = line.split()
    for word in word_list:
        print(word)
infile.close()
'''
'''
#p326

infile=open("d:/before.txt","r")
outfile = open("d:/after.txt","w")

s=infile.read()

outfile.write(s)

infile.close()
outfile.close()

'''
'''
#328, 행맨게임

#맞으면 맞춘 글자가 보이고, 틀리면 시도횟수만 하나 증가한다.(맞춰도 시도횟수 카운트는 올라간다.)
#행맨의 단어는 words.txt 파일에서 불러온다
#랜덤으로 단어 중에 하나를 가져오고, 각 단어에 대해서 앞에서 부터 한 글자씩 비교한다.
#그래서 한 번이라고 글자가 맞으면 그글자는 글자가 보이게끔 하고, 아직 맞추지 못한 글자는 그대로 출력한다.
#모든 글자가 밝혀지면 지금까지 마쳤다는 신호와 지끔까지 시도횟수를 보여준다.

import random

#파일생성.

outfile=open("d:/words.txt","w")
outfile.write("apple \n")
outfile.write("wonderful \n")
outfile.write("approve \n")
outfile.write("tablet \n")
outfile.close()

infile = open("d:/words.txt","r")
lines = infile.readlines() #리스트 형태로 들어감

word = random.choice(lines).strip()
print(word)
count=0
guess=""
question="_"*len(word) #문제를 몇 글자인지 _(언더바)로 표시
print("%s %d글자입니다."%(question,len(question)))
while guess!=word:
    guess=input("한글자씩 맞춰보세요!(알파벳) : ")
    for w in range(len(word)):
        if word[w]==guess:
            temp_l=question[:w]
            temp_r=question[w+1:]
            question=temp_l+guess+temp_r
    if guess==word:
        question=word
    print(question)
    count +=1
    print("시도횟수 %d"%count)

print("도전완료! 정답은 %s" %question)
print("총 시도횟수 %d!!"%count)
'''
'''
#330
gameOption = {"Sound":8, "VideoQuality ":"HIGH", "Money":100000,"WeaponList":["gun","missile","knife"]
    }
import pickle

file = open("d:/save.p","w")
pickle.dump(gameOption,file)
file.close()
'''
'''
#332, 이진파일로 저장
gameOption = {"Sound":8, "VideoQuality ":"HIGH", "Money":100000,"WeaponList":["gun","missile","knife"]
    }
import pickle

file = open("d:/save.p","wb")
pickle.dump(gameOption,file)
file.close()
'''
'''
#333, 이진파일 불러오기_로드

import pickle

file = open("d:/save.p","rb")

obj = pickle.load(file)
print(obj)
file.close()
'''
'''
#334 패스, chapter 10 하고 해야함.
'''

#338, 연습문제
'''
#1번

#사용자가 입력한 텍스트 파일을 열어서 파일 안에 글자가 몇 글자나 있는지를 계산하라.

#임의 변수로 파일의 위치를 입력받는다.
#해당 파일을 읽어와서 한글자씩 분할한다. 그리고 한글자를 분할할 때마다 cnt 값을 올려준다.
#모든 과정이 끝나고 몇글자인지 나온 값을 출력한다.

outfile = open("d:/chapter11-1.txt","w")
x=input("원하시는 단어 및 문장을 입력해주세요 \n")
outfile.write(x)
location = input("파일의 위치를 입력해주세요:")
cnt=0
outfile.close()

infile = open(location, "r")
lines=infile.readlines()#각 줄을 리스트의 형태로 저장
print(lines)
for words in lines:
    for word in words.strip():
        cnt+=1
print("총 단어의 개수는 %d 입니다."%cnt)
'''

#2번
#사용자로부터 파일 이름과 삭제할 문자열을 입력받는다. 파일을 열어서 사용자가 원하는 문자열을 삭제한 후에 다시 파일을 쓴다.
'''
file = open("d:/chapter11-2.txt","w+")
file.write("ppap apple pine")
file.close()

file = open("d:/chapter11-2.txt","r")
clear=""
lines=file.readlines()
print(lines)
file.close()
guess=input("찾는 단어는?? :")

for l in range(len(lines)):
    line=lines[l]
    words = line.split()
    for w in range(len(words)):
        if words[w]!=guess:
            clear +=words[w]+" "
    clear+="\n"
    
file = open("d:/chapter11-2.txt","w+")
file.write(clear)
file.close()
'''
#2번_2
'''
file = open("d:/chapter11-2_2.txt","w")
file.write("all is my life")
file.write("is beautiful.")
file.write("until you die")
file.close()

file= open("d:/chapter11-2_2.txt","r")
x=file.read()

print(x)

file.close()

guess=input("찾는 단어는?? :")
x=x.replace(guess,"")

file = open("d:/chapter11-2_2.txt","w")
print(x)
file.write(x)
file.close()
'''
#3번
'''
#사용자가 입력하는 파일에 있는 각 문자들이
#나타내는 빈도를 계산하는 프로그램을 작성하라.
file = open("d:/chapter11-3.txt","w")
file.write("all is my life \n")
file.write("is beautiful. \n")
file.write("until you die \n")
file.close()

dic = {}
file = open("d:/chapter11-3.txt","r")
lines=file.read()
for line in lines:
    line=line.strip()
    #print(type(lines))
    for s in line:
        if s not in dic.keys():
            dic[s]=1
        else:
            dic[s]+=1
print(dic)
file.close()
#print(lines)
'''
            
#4번

#파이썬은 객체를 파일에 저정할 수 있다. pickle 모듈을 사용하여 정수12, 실수 3.14, 리스트[1,2,3,4,5]를 이진파일 "test.dat"에
#저장하였다가, 다시 읽는 프로그램을 작성하고 테스트하라.
'''
import pickle

i = 12
f  = 3.14
l = [1,2,3,4,5]

#name="alex"
#level=12
#dic ={"name":name,"level":level}
list_value=[12,3.14,[1,2,3,4,5]]

file = open("d:/test.dat","wb")
pickle.dump(list_value[0],file)
pickle.dump(list_value[1],file)
pickle.dump(list_value[2],file)
file.close()

file = open("d:/test.dat","rb")
p=pickle.load(file)
print(p)
p=pickle.load(file)
print(p)
p=pickle.load(file)
print(p)

file.close()
'''
#5번

#텍스트 파일 "data.txt"에 실수값들이 저장되어 있다고 한다. 한 줄에 하나의 실수만 저장 되어 있다고 한다.

file = open("d:/chapter11-5.txt","w")
file.write("17.0 \n")
file.write("51.0 \n")
file.write("68.5 \n")
file.write("82.9 \n")
file.write("103.2 \n")
file.close()

file=open("d:/chapter11-5.txt",)
values=file.readlines()
file.close()
sum=0
for v in values:
    sum+=float(v)
print(sum)
print(float(sum/len(values)))

#6번
#Tkinter
