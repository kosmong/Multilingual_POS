import Read_Data as rd
import pandas as pd
import nltk

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
