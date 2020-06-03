#PS4Q3
#Write a function that accepts a hyphen-separated sequence
#of words and then outputs/prints the same hyphen separated
#sequence but with all the words sorted alphabetically.
#Take input from the user.
#Eg.
#Input: boy-apple-giraffe-chipotle-stroopwafel
#Output: apple-boy-chipotle-giraffe-stroopwafel


#ALGORITHM
#take string input
#split on the '-' dash...store in list
#def....function....takes one list
    #.sort the list..
    #.join the list using "-"...return this string
#call function...pass user string...print what you get in return






















#code
sep= "-"
wordsList = input("Type hypen separated sentence").split(sep)

def hyphenWords(list1):
    wordsList.sort()
    return(sep.join(wordsList))

print(hyphenWords(wordsList))

#suggestion: think about how you would do it if there wasn't any sort function
#available
#hint: you'll need two loops -->nested Loops.

        
