#S11Q4
#Download the text file:
#https://drive.google.com/file/d/1RG_29OiMY9KmKjWDW5Gq1nzzEzYNJG4G/view?usp=sharing 
#This is Ken’s marksheet. 
#Write a program to kind the subject in which he got
#the highest marks and print that subject.

#Output:
#“Ken’s best performance is in the subject Geography.”





#ALGORITHM
#open file to read
#read lines....so read each line into separate list element
#create temp variable maxVal to store maximum value...put 0
#create variable to name of subject with highest marks. ...put random subj

#start a loop...loop over each element of list..starting from index 2
#the first two elements are useless...hence we are skipping them
    #split the element on ": "
        #store both in list.
        #if second element of this list is greater than max val
            #then store the first element ...subj name in subj variable


#print the final value in max subj variable













#CODE
f = open("kenSmithMarksheet.txt",'r')
maxVal,maxSubj = 0,"Mathematics" #assume some subject and marks 

listOfLines = f.read().split("\n")
del(listOfLines[-1]) #check output after the above statement
#there's an empty string as the last element.
#delete such spurious input...it affects generalization and looping

for everyElement in listOfLines[2:]:
    tempList = everyElement.split(": ")
    if int(tempList[1]) > maxVal:
        maxVal = int(tempList[1])
        maxSubj = tempList[0]

print("Ken’s best performance is in the subject",maxSubj, ".")


