#S11Q7
#Write a program that takes a string as input and
#generates a text file order.txt having on each line a
#number and all the words that have the length equal to
#that number, ordered by their first letter. 
#If there are no words of that length the number won't be added to the file.

#Input:
#" The sun is shining bright today.
#Is such a wonderful day to go out and have a walk in the park."

#Output - Contents of order.txt should be.
#1 a a
#2 go is in IS to
#3 and day out The the sun
#4 have park such walk
#5 today
#6 bright
#7 shining
#9 wonderful






#ALGORITHM
#take string input...replace the full stops with space..and split on space
#(we do this so that we don't get full stops in final answer)
#find biggest word, smallest word
#create empty list ...temp variable..

#start a loop that goes from length of smallest word to length of biggest word
    #start a loop that goes over every word
        #check if length of word is same as initial loop iterator....
            #if yes then add it to the temp write list
    #if this write list is not empty...then write it to file...in proper format
    #empty write list and set it to [] again.

#close the file
















#CODE
wordsList = (input("Type your sentence").replace("."," ")).split(" ")
wordsList = list(filter(None,wordsList))
maxWord,minWord = max(wordsList), min(wordsList)
writeList = []

f = open("order.txt",'w')

for i in range(len(minWord),len(maxWord)):
    for everyWord in wordsList:
        if len(everyWord) == i:
            writeList.append(everyWord)
    if writeList != []: f.write(str(i) + " " + " ".join(writeList) + "\n")
    writeList = []

f.close()
