'''
1.) Encipher 
  a.) Get any single string
  b.) Split them by each single letter, character or space
  c.) Coerce them into number using ord()
  d.) Use simple algebraic function: 7(x/10) = y
  e.) Coerce them into enciphered letters using chr()
  f.) Reverse the list
  g.) Coerce the list into string

2.) Decipher
  a.) Coerce string into the list
  b.) Reverse the list
  c.) Coerce the list into numeric list
  d.) Use the reverse algebraic function: x = 10*y/7
  e.) Use chr(), an inverse of ord()
  f.) Group all letters, characters, and spaces
'''

def encipher(n):                       # Create the encipher function
    if type(n) != type('true?'):
        print('Please add the string') # If this data is not string type.
    else:
        splstr = list(n)             # Coerce string into list of characters
        thelist = []
        for i in range(len(list(n))):# Coerce list into numeric list using algebraic function
            x = ord(list(n)[i])
            x = int(10*x/2)
            if x >= 1114111:         # In case if someone wants to mess around with this cipher
                x = '􏿿'
            else:
                x = chr(x)
            thelist.append(x)
        thelist.reverse()
        enciphered = "".join(thelist)
        return enciphered

def decipher(n):                       # Create the decipher function
    if type(n) != type('true?'):
        print('Please use the enciphered string') # If this data is not string type.
    else:
        splstr = list(n)             # Coerce string into list of characters
        thelist = []
        for i in range(len(list(n))):# Coerce list into numeric list using algebraic function
            x = ord(list(n)[i])
            x = int(2*x/10)
            x = chr(x)
            thelist.append(x)
        thelist.reverse()
        deciphered = "".join(thelist)
        return deciphered

txt = "Usually, the solitude is often pleasant to have, staying away from the people capable of baning my existence. But since March 2020, everything is changed, nay transmorgified. Here is nothing but nighmarish. Being isolated is hell. Perhaps, to you, being isolated at home for one year seems easy. Being isolated for one year equals 4 million years. That is, the solitude turns into loneliness."

entxt = encipher(txt)
detxt = decipher(entxt)

print(entxt)
print("")
print(detxt)
