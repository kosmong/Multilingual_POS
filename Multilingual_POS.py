import Read_Data as rd
import pandas as pd
import nltk
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline

CSV_FOLDER = "Prat data/tagged_interviews_csv/"


# REQUIRES: the symbol always signify a disfluency and never appears by itself
# might be better to make own tokenizer
# TODO: refactor so that it put the disfluencies together and recombine symbols
# token will be tagged tokens, list of pairs (word, tag)
# if it is &, combine with next word and change tag to DIS (dispfluencies)
# if it is @, combine with next word and change tag to LABL (label)
# if it is part of the disfluency list, change tag to DIS
def recombine_and_retag(tagged_tokens, recombine, disfluencies):
    to_pop = []
    out_tagged = tagged_tokens.copy()
    for i in range(len(tagged_tokens)):
        wrd = tagged_tokens[i][0]
        pos = tagged_tokens[i][1]

        if wrd in recombine:
            recombined_wrd = tagged_tokens[i][0] + tagged_tokens[i + 1][0]
            to_pop.append(i + 1)

            if wrd in disfluencies:
                out_tagged[i] = (recombined_wrd, 'DIS')
            else:
                out_tagged[i] = (recombined_wrd, 'LABEL')

        if wrd in disfluencies and wrd not in recombine:
            out_tagged[i] = (wrd, 'DIS')

    for p in range(len(to_pop)):
        # have to account for shrinking array of tokens
        # the position in the earlier will always be smaller
        # as we remove an element from the left, the index for right elements-1
        # p is the number of elements we have already popped
        out_tagged.pop(to_pop[p] - p)

    return out_tagged


def recombine_hugging(tagged_tokens):
    """ Take the hugging face tagged tokens and recombine the split up subwords (with ##) to 1 word if they have same POS
        remove extra words"""
    cleaned = []
    prev_word = ''
    prev_tag = ''
    to_pop = []

    for i in range(len(tagged_tokens)):
        entry = tagged_tokens[i]
        tag = entry['entity']
        word = entry['word']

        # if current word is a sub word (contains ##) and has the same POS, recombine with previous word
        # thus, the pos that we need to remove is i-1 (previous)
        if "##" in word and prev_tag == tag:
            word = prev_word + word.replace("##", '')
            to_pop.append(i-1)

        cleaned.append((word, tag))
        prev_word = word
        prev_tag = tag

    for p in range(len(to_pop)):
        # have to account for shrinking array of tokens
        # the position in the earlier will always be smaller
        # as we remove an element from the left, the index for right elements-1
        # p is the number of elements we have already popped
        cleaned.pop(to_pop[p] - p)

    return cleaned


# Takes in a sentence, tokenize and does initial POS tagging
def POS_sentence(sentence: str):
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words = set(stop_words)
    stop_words.add('mmm')
    tokens = nltk.word_tokenize(sentence)
    # recombine &, @ and the utterance right after
    # TODO: uncomment after done or take out
    # tokens = recombine_and_retag(tokens, ["&", "@", "＠"])
    # relabelled disfluencies as DIS
    disfluencies = ["&", "mmm", "mnmm", "mmhm", "mm", "ahh", "huh", "umm"]
    cleaned = [w for w in tokens if w not in stop_words]
    cleaned_tagged = nltk.pos_tag(cleaned)
    tagged = nltk.pos_tag(tokens)
    return tagged, cleaned_tagged


# store the tagged parsed sentences in panda dataframe
# save as csv into folder
# structure: [tags:, words:]
def textgrid_into_csv(name: str, tier):
    sym = ["&", "@", "＠"]
    disf = ["&", "mmm", "mnmm", "mmhm", "mm", "ahh", "huh", "umm", "mhm", "um"]
    interview_ans = []

    for interval in tier.get_intervals():
        sentence = interval.get_text()
        if sentence != '':
            tagged, _ = POS_sentence(sentence)
            tagged = recombine_and_retag(tagged, sym, disf)
            wrds, tags = zip(*tagged)
            tagged_df = pd.DataFrame({'POS tags': tags, 'Words': wrds})
            interview_ans.append(tagged_df)

    answers_df = pd.concat(interview_ans)
    answers_df.to_csv(CSV_FOLDER + name + '.csv', index=False)


# create a dataframe that has words and their POS tag as axis labels
# 2D frame, increment when an instance of word with POS tag is seen
# use it to calculate probabilities
# TODO: change this so that it grows a List instead of dataframe
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


# make code switch counted dataframe, maybe change the master dataframe too
# dataframe structure: [Word, [(tag, count)]]
def make_csw_count(mcount_df: pd.DataFrame):
    csw_counts_dict = {'Words': [], "Count per POS tags": []}
    # first get all unique words
    words = mcount_df['Words'].unique().tolist()
    for word in words:
        word_df = mcount_df.loc[mcount_df['Words'] == word]

        wrd_pos_count = []
        for row in word_df:
            wrd_pos_count.append((row[1], row[3]))

        csw_counts_dict['Words'].append(word)
        csw_counts_dict['Count per POS tags'].append(wrd_pos_count)

    csw_df = pd.DataFrame(csw_counts_dict)
    return csw_df


# TODO: need to make comparisons between the different textgrids
# potential problems:
# 1. the tokenizations are different for each, how do we compare?
#           we can do regular expression matching? first find matching word,
#           then find ends of () and get what is between
# TODO: the file representations are invalid in the ones saved from prat? why? perhaps change encoding to utf-8
def find_text_diff(model_generated:str, gold_standard:str):
    # first find number of words and if it is the same for both
    model_split = model_generated.split(') ')
    golden_split = gold_standard.split(') ')
    len_m = len(model_split)
    len_g = len(golden_split)
    print(len_m, len_g)



model = rd.ParsedTextgrid('Prat data/tagged_interviews_textgrid/POS_VM24A_English_I1_20181209.TextGrid')
# gold = rd.ParsedTextgrid('/Users/kosmong/Desktop/Cogs 402/Multilingual_POS/Prat data/corrected_tagged_textgrid/old_POS_VM24A_DT_Correct_English_I1_20181209_DT_Edited.TextGrid')
gold = rd.ParsedTextgrid('Prat data/corrected_tagged_textgrid/POS_VM24A_DT_Correct_English_I1_20181209_DT_Edited.TextGrid')
model_generated = model.get_tiers()[2].get_intervals()[3].get_text()
gold_standard = gold.get_tiers()[2].get_intervals()[3].get_text()
print(model_generated, gold_standard)
find_text_diff(model_generated, gold_standard)

# make a new multilevel conditional random field
# with variable length features that detect, word, phrasal (NP, VP), sentence


# # trying Hugging face
# model_name = "QCRI/bert-base-multilingual-cased-pos-english"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForTokenClassification.from_pretrained(model_name)
#
# pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
# # outputs = pipeline("A test example")
# # print(outputs)
# longer_sen = ("mmm oh i also learned the 古箏 like yeah. "
#                       "i don't know like i was like just putting it out there like just cuz we were on the topic and i was like i totally forgot yeah. "
#                       "cuz at like temple they had like 古箏 like they had this 古箏 teacher that volunteered to like teach us so i learned 古箏 for a while i have 古箏 at home and like. "
#                       "yeah it's like it's pretty fun actually um uh and any instrument that i want to play okay if i ever wanted to try something i'd probably try like the tuba or something. "
#                       "just cuz like everything i've played was like strings related and stuff but i've never tried like like a wind instrument. "
#                       "but i know also like controlling my breath is probably pretty hard cuz i don't know i know you have to have like the um embrasure or something like that. "
#                       "for any like like those weird instruments but a tuba would be cool")
#
# out1 = pipeline(longer_sen)
# tagged_pairs = []
# for entry in out1:
#     tagged_pairs.append((entry['entity'], entry['word']))
#
# print(tagged_pairs)

