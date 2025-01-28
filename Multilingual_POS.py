import Read_Data as rd
import pandas as pd
import nltk


CSV_FOLDER = "Prat data/tagged_interviews_csv/"
TEXTGRID_FOLDER = "Prat data/interview_textgrids_iu_and_cs_intervals/"
FILES_NAMES = ["VF19A_English_I1_20181114",
               "VF19B_English_I1_20190213",
               "VF19C_English_I1_20190224",
               "VF19D_English_I2_20190308",
               "VF20A_English_I2_20181119",
               "VF20B_English_I2_20181203",
               "VF21A_English_I1_20190130",
               "VF21B_English_I2_20190204",
               "VF21C_English_I2_20190211",
               "VF21D_English_I1_20190306",
               "VF22A_English_I2_20181206",
               "VF23B_English_I1_20190121",
               "VF23C_English_I2_20190128",
               "VF26A_English_I2_20190303",
               "VF27A_English_I1_20181120",
               "VF32A_English_I2_20190213",
               "VF33B_English_I1_20190206",
               "VM19A_English_I1_20191031",
               "VM19B_English_I2_20191104",
               "VM19C_English_I1_20200211",
               "VM19D_English_I2_20200211",
               "VM20B_English_I1_20181126",
               "VM21A_English_I1_20181206",
               "VM21B_English_I1_20190313",
               "VM21C_English_I2_20190403",
               "VM21D_English_I2_20200309",
               "VM21E_English_I2_20200309",
               "VM22A_English_I2_20181210",
               "VM22B_English_I1_20200309",
               "VM23A_English_I1_20200310",
               "VM24A_English_I1_20181209",
               "VM25A_English_I1_20190923",
               "VM25B_English_I1_20200224",
               "VM34A_English_I2_20191028"]


def POS_sentence(sentence: str):
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words = set(stop_words)
    stop_words.add('mmm')
    tokens = nltk.word_tokenize(sentence)
    cleaned = [w for w in tokens if w not in stop_words]
    cleaned_tagged = nltk.pos_tag(cleaned)
    tagged = nltk.pos_tag(tokens)
    return tagged, cleaned_tagged


# test out how nltk work
simple_eng = "I went to the mall."
tagged_eng, cleaned_eng = POS_sentence(simple_eng)

# test for canto
simple_canto = "古箏"
tagged_simple, cleaned_simple = POS_sentence(simple_canto)

# test for code switched sentence
csw_sen = 'i like learned 古箏 for like a while'
tagged_csw, cleaned_csw = POS_sentence(csw_sen)
csw_wrds, csw_tags = zip(*tagged_csw)
csw_df = pd.DataFrame({'POS tags': csw_tags, 'Word tokens': csw_wrds})

csw_sen2 = 'they had this 古箏 teacher that volunteered to like teach us'
tagged_csw2, cleaned_csw2 = POS_sentence(csw_sen2)
csw_a2, csw_b2 = zip(*tagged_csw2)
csw2_df = pd.DataFrame({'POS tags': csw_a2, 'Word tokens': csw_b2})
df = pd.concat([csw_df, csw2_df])

longer_sen = ("mmm oh i also learned the 古箏 like yeah. "
              "i don't know like i was like just putting it out there like just cuz we were on the topic and i was like i totally forgot yeah. "
              "cuz at like temple they had like 古箏 like they had this 古箏 teacher that volunteered to like teach us so i learned 古箏 for a while i have 古箏 at home and like. "
              "yeah it's like it's pretty fun actually um uh and any instrument that i want to play okay if i ever wanted to try something i'd probably try like the tuba or something. "
              "just cuz like everything i've played was like strings related and stuff but i've never tried like like a wind instrument. "
              "but i know also like controlling my breath is probably pretty hard cuz i don't know i know you have to have like the um embrasure or something like that. "
              "for any like like those weird instruments but a tuba would be cool")
tagged_long, cleaned_long = POS_sentence(longer_sen)
a, b = zip(*tagged_long)


# store the tagged parsed sentences in panda dataframe
# save as csv into folder
# structure: [tags:, words:]
def textgrid_into_csv(name: str, tier):
    interview_ans = []
    for interval in tier.get_intervals():
        sentence = interval.get_text()
        if sentence != '':
            tagged, cleaned = POS_sentence(sentence)
            wrds, tags = zip(*tagged)
            tagged_df = pd.DataFrame({'POS tags': tags, 'Words': wrds})
            interview_ans.append(tagged_df)

    answers_df = pd.concat(interview_ans)
    answers_df.to_csv(CSV_FOLDER + name + '.csv', index=False)


# parse textgrid, store as dataframe, save as csv file
for filename in FILES_NAMES:
    path = TEXTGRID_FOLDER + filename + ".TextGrid"
    parsed_textgrid = rd.ParsedTextgrid(path)
    convenience_tier = parsed_textgrid.get_tiers()[2]
    assert convenience_tier.get_name() == "convenience-IU"
    textgrid_into_csv(filename, convenience_tier)



