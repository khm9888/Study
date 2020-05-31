#S11Q5
#Download the text file:
#https://drive.google.com/file/d/1hdrlJdKq57jRPmFTwDFAOpGPDNHZpsL-/view?usp=sharing 
#Count and print the number of vowels and spaces in this text file.






#ALGORITHM
#open file for reading...
#create count variable ..set to 0
#use .read...read everything as single string
#start loop. ...iterate over string...
    #if character is any of 'a' 'e' 'i' 'o' 'u'
        #increment count

#print final value of count


















#CODE
f = open("pollution.txt",'r')
count = 0
fileText = f.read()

for everyChar in fileText:
    if everyChar in ['a','e','i','o','u']:
        count+=1
print("No of vowels in this file:",count)
