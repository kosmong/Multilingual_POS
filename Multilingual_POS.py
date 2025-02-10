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


# REQUIRES: the symbol always signify a disfluency and never appears by itself
# might be better to make own tokenizer
def recombine(tokens, symbols):
    to_pop = []
    for i in range(len(tokens)):
        if tokens[i] in symbols:
            tokens[i] = tokens[i] + tokens[i + 1]
            to_pop.append(i + 1)

    for p in range(len(to_pop)):
        # have to account for shrinking array of tokens
        # the position in the earlier will always be smaller
        # as we remove an element from the left, the index for right elements-1
        # p is the number of elements we have already popped
        tokens.pop(to_pop[p] - p)

    return tokens


def POS_sentence(sentence: str):
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words = set(stop_words)
    stop_words.add('mmm')
    tokens = nltk.word_tokenize(sentence)
    # recombine &, @ and the utterance right after
    tokens = recombine(tokens, ["&", "@", "＠"])
    cleaned = [w for w in tokens if w not in stop_words]
    cleaned_tagged = nltk.pos_tag(cleaned)
    tagged = nltk.pos_tag(tokens)
    return tagged, cleaned_tagged


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
# for filename in FILES_NAMES:
#     path = TEXTGRID_FOLDER + filename + ".TextGrid"
#     parsed_textgrid = rd.ParsedTextgrid(path)
#     convenience_tier = parsed_textgrid.get_tiers()[2]
#     assert convenience_tier.get_name() == "convenience-IU"
#     textgrid_into_csv(filename, convenience_tier)


# create a dataframe that has words and their POS tag as axis labels
# 2D frame, increment when an instance of word with POS tag is seen
# use it to calculate probabilities
def make_count_df(labelled_df: pd.DataFrame, count_df: pd.DataFrame):

    for row in labelled_df.itertuples():
        pos_tag = row[1]
        word = row[2]

        # if already contains POS Word combo, increment count
        # find row that contains the POS tag and word pair
        found_i = count_df.loc[(count_df['POS tags'] == pos_tag) & (count_df['Words'] == word)].index
        if found_i.empty:
            count_df.loc[len(count_df)] = [pos_tag, word, 1]
        else:
            count_df.loc[found_i, 'Count'] = count_df.loc[found_i, 'Count'] + 1

    return count_df

# # make master count df
# columns = ['POS tags', 'Words', 'Count']
# m_count_df = pd.DataFrame(columns=columns)
# for filename in FILES_NAMES:
#     path = CSV_FOLDER + filename + ".csv"
#     df = pd.read_csv(path)
#     m_count_df = make_count_df(df, m_count_df)
#
# m_count_df.sort_values(by='Words', inplace=True, ascending=False)
# m_count_df.to_csv(CSV_FOLDER + 'Master_Counts.csv')


# test if sentences with no disfluencies get wrongly classified
example_canto = "楊枝甘露 is delicious"
tagged_ex, _ = POS_sentence(example_canto)
print(tagged_ex)

# I think this is correct
tagged_canto2, _ = POS_sentence("my favourite is still 黑芝麻")
tagged_eng, _ = POS_sentence("my favourite is still sesame")
print(tagged_canto2)
print(tagged_eng)


# & means discontinuations, must stay with the next word (done)
# check unique words and maybe hand tag
# discontinuations and disfluencies: false start (&) um, ahhs, mmm
# cantonese testing, where is the pos different?
# @ sign for code-switching: language other than english or canto
# @m: mandarin, @j: japanese, @ml: malay and @i: indonesian; 21a, 21c, 22a, 24a, 25a, 27a
# what is @h? m25A

# check where the modifiers for noun and adjective
# 黑芝麻 is wrong F32A, maybe correct actually
# Definitely wrong
# 楊枝甘露 is labelled as verb F27A
# this is also pre warned with can I speak in cantonese
# In context, I understand why it is a verb;
# 19b 2692: nihao

# curious about read speech, no disfluencies, and spontaneous speech aspect
# I think maybe the bigger problem is the spontaneous speech aspect!
# need to compare read speech codeswitch to spontaneous speech codeswitch
# help actually there is a lot of mislabelling

# get disfluencies relabelled (mmm, umm, xxx)
# how about repeated words?
# how much does disfluencies and repeated words predict codeswitching??
# Is it just normal disfliencies?
# deep seek: how does that work
# learning the data