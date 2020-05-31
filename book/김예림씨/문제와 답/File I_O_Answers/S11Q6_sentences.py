#S11Q6
#Download the text file: 
#https://drive.google.com/file/d/1u4QsRoJJi9zIrShZBeYhKwuqhBpzCgxu/view?usp=sharing 
#Count and print the number of sentences in this file. Sentences, NOT lines.

#ALGORITHM

#open the file
#read the lines..strip the final new line and split on "."
#now use filter to filter out any empty strings if there...
#print the length of the final list...this is the true no of sentences




















#CODE
f = open("programmer.txt",'r')

sentencesList = f.read().rstrip("\n").split(".")
#print(sentencesList)
#if you print the list now...you'll see a problem...
#there's an empty string at the very end
#this happens due to the split operation we did above.

sentencesList = list(filter(None,sentencesList))
#in this line we filter out the empty string
#we use the filter function on the list sentencesList
#filter gives us a way of extracting all the items that fulfill a certain
#condition

#in this case we use the condition as None
#when the condition is None...it means extract all elements that are true
#empty strings are False...so they don't get extracted.

print(sentencesList)
print("No of lines in this file are",len(sentencesList))
