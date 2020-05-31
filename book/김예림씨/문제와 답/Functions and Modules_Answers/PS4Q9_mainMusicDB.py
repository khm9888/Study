#PS4Q9
#music album database


#algorithm

#--main code

#import db functions module
#set db length to 5

#define user input function...takes one integer (db legth)
    #create empty dictionary..
    #iterate till range (dblength)
        #call store new entry function, pass the same empty dicitonary to it
        #store the return in same db again
        #(so everytime in this loop, this intruction calls the store
        #(entry function...takes a new entry and UPDATES it in the same
        #(DB. we call it five times since we want five entries)

#call take input function...pass db length to it.
#it will return the final music dictionary...catch this!

#ask user for album name

#call find entry function...pass query user just gave
#function will return true or false based on search result...catch this!

#if return value was false...
    #call the entry not found function...pass the user query again here



#---------------
#----MODULE ALGO

#def store new entry function...it will take a dictionary
    #take 3 inputs from user. all strings.. album, artist, year
    #put them in the dictionary...remember to use album is key
    #(if you see question, that what we would be asked to search on
    #(we can then put artist, year as list
    #return this dictionary

#def find entry function...takes user query so string and the dictionary
    #if the query string is in any of the keys of the dictionary
    #(remember album names are keys)
        # print the success found string
        #return true
    #else
        #return false


#def the entry not found function...it will take a string...userQuery
        #print the not found string
        #return (nothing particular to return)
    






















#code
import databaseFunctions

databaseLength = 5

def takeUserInput(dbLength):
    db = {}
    for i in range(dbLength):
        db = databaseFunctions.storeNewEntry(db)
    return db

musicDict = takeUserInput(databaseLength)


userQuery = input("Enter the album you want to search for in database")

queryStatus = databaseFunctions.findEntry(userQuery,musicDict)

if queryStatus == False:
    databaseFunctions.entryNotFound(userQuery)
