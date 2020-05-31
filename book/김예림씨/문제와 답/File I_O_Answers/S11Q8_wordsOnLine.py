#S11Q8
#Write a program that takes a string as an input and
#generates a text file wordsOnLine.txt having on each line
#as many words as the number of the line in the text.

#Input
#" All the things we could learn, if only we would be patient enough to keep trying more every time we fail. "

#Output - Contents of wordsOnLine.txt should be
#All
#the things
#we could learn
#if only we would
#be patient enough to keep
#trying more every time we fail






#ALGORITHM
#take input...split...store in list1 variable

#set count = length of list1

#open file in which you want to write

#start loop...temp variable i iterate over value of count...range(count):
    #start loop...temp variable j iterate over value of i (from above):
        #if...check if list is empty:
            #if yes just put 'continue',don't do anything,just skip the iteration
        #else...
            #make the string you want to write ...1st element of list1 + space
            #write this to file
            #del the 1st element of list1...the one you wrote just now.

    #write a new line to the file

#close the file


















#CODE
list1 = input("type sentence").split()

#list1 = ['this', 'is', 'sentence', 'stop']
count = len(list1)
f = open("wordsOnLine.txt.",'w')

for i in range(count):
    for j in range(i):
        if list1 == []:
            continue
        else:
            f.write(str(list1[0]) + " ")
            del list1[0]
    f.write("\n")

f.close()
