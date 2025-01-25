import Read_Data as rd
import nltk


# test out how nltk work
sen = "I went to the mall."
token = nltk.word_tokenize(sen)
# print(token)
tagged = nltk.pos_tag(token)
# print(tagged)

code_switched_sen = 'i like learned 古箏 for like a while'
token_csw_sen = nltk.word_tokenize(code_switched_sen)
# print(token_csw_sen)
tagged_csw = nltk.pos_tag(token_csw_sen)
# print(tagged_csw)

longer_sen = ("mmm oh i also learned the 古箏 like yeah. "
              "i don't know like i was like just putting it out there like just cuz we were on the topic and i was like i totally forgot yeah. "
              "cuz at like temple they had like 古箏 like they had this 古箏 teacher that volunteered to like teach us so i learned 古箏 for a while i have 古箏 at home and like. "
              "yeah it's like it's pretty fun actually um uh and any instrument that i want to play okay if i ever wanted to try something i'd probably try like the tuba or something. "
              "just cuz like everything i've played was like strings related and stuff but i've never tried like like a wind instrument. "
              "but i know also like controlling my breath is probably pretty hard cuz i don't know i know you have to have like the um embrasure or something like that. "
              "for any like like those weird instruments but a tuba would be cool")
token_long = nltk.word_tokenize(longer_sen)
print(token_long)
tagged_long = nltk.pos_tag(token_long)
print(tagged_long)

# remove stop words, and filler words such as "like" and "mmm"
stop_filler = nltk.corpus.stopwords.words('english')
stop_filler = set(stop_filler)
stop_filler.add('mmm')
# print(stop_filler)
# print(type(stop_filler))
longer_sen_cleaned = [w for w in token_long if w not in stop_filler]
print(longer_sen_cleaned)
tagged_long_cleaned = nltk.pos_tag(longer_sen_cleaned)
print(tagged_long_cleaned)