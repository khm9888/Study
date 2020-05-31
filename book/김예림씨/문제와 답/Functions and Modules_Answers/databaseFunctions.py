#module PS4Q9
def storeNewEntry(dbDict):
    artist = input("Give artist")
    albumName = input("Give name of music album")
    yearOfRelease = input("What's the year of release?")
    dbDict[albumName] = [artist,yearOfRelease]
    return dbDict


def findEntry(str1,dbDict):
    if str1 in dbDict.keys():
        print("""The Artist who created this album is,""", dbDict[str1][0],""". They released it in """, dbDict[str1][1], """.""")
        return True
    else:
        return False

def entryNotFound(str1):
    print("""Error Statement : The album, """, str1, """wasn't found in our database. Please check your entry again.""")
    return

