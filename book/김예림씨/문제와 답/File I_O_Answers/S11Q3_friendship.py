#S11Q3
#Write a program that takes a string as input and
#generates a text file friendship.txt containing only the
#letters in that string.

#Input
#""" 86I20 ca88nn42ot w8856ai77t t22o s7p74e58n1d
#t1i0m3e8 w698it55h m22y b2e338st fr88i5e9n2ds."""

#Output - Contents of friendship.txt should be:
#I cannot wait to spend time with my best friends.



#ALGORITHM
#take user string...split on spaces..store in list
#create a temp string variable...to store the extracted word later

#start a loop over each element of this list
    #now loop over each word...so one character at a time
        #if character is letter
            #add that to the temp word
    #add that word to the former list

#delete the elements apart from the extracted words...so delete all the previous
#non-extracted words.

#join the words in the list with space and write to file
#close file















#CODE
list1 = input("Type the sentence with numbers").split()
tempWord = ""
list2 = []

for everyWord in list1:
    for everyLetter in everyWord:
        if (everyLetter.isalpha()) or (everyLetter =="."):
            tempWord += everyLetter
    list2.append(tempWord)
    tempWord = ""

finalString = " ".join(list2)

f = open("friendship.txt",'w')
f.write(finalString)
f.close()

        
