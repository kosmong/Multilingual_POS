import numpy as np

import Read_Data as rd
import pandas as pd
import nltk
import string
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import sklearn_crfsuite
from sklearn_crfsuite import metrics

CSV_FOLDER = "Prat data/tagged_interviews_csv/"
lower = string.ascii_lowercase
upper = string.ascii_uppercase
symbols = []
# TODO more place names:
PLACE_NAMES = ["hong kong", "vancouver", "toronto", "japan", "canada", "greater vancouver area", "tokyo", "nagano",
               "hiroshima", "miyajima", "iwakuni", "osaka", "langara", "mong kok", "ubc", "bcit", "sfu",
               "montreal", "burnaby", "mcgill", "ut", "china", "singapore", "ontario", "queens"]


# REQUIRES: the symbol always signify a disfluency and never appears by itself
def recombine_and_retag(tagged_tokens, recombine, disfluencies):
    """token will be tagged tokens, list of pairs (word, tag)
    if it is &, combine with next word and change tag to DIS (dispfluencies)
    if it is @, combine with next word and change tag to LABL (label)
    if it is part of the disfluency list, change tag to DIS"""
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
            to_pop.append(i - 1)

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


# TODO: file representations can be changed in pycharm
# TODO: find how far disfluency or repeated words is from a code switch
def find_text_diff(model_generated: str, gold_standard: str):
    # first find number of words and if it is the same for both
    # model_split = model_generated.split(') ')
    # gold_split = gold_standard.split(') ')
    # len_m = len(model_split)
    # len_g = len(gold_split)

    # # turn the word and tag back into pairs
    # model_pairs = tagged_words_to_pairs(model_split)
    # gold_pairs = tagged_words_to_pairs(gold_split)
    model_pairs = model_generated
    gold_pairs = gold_standard
    len_m = len(model_pairs)
    len_g = len(gold_pairs)
    # assume that human tagged data will put words together such as names
    # so will have less or equal words
    if len_g > len_m:
        raise UnexpectedTokenizationException(gold_pairs, model_pairs)

    # for each pair, compare word
    # assume gold standard does not tokenize incorrectly
    # TODO: we probably want to have a df to store the counts and also what words are split incorrectly
    correct = 0
    incorrect = 0
    bad_split = 0
    total_offset = 0
    code_switch_count = 0
    repeated_count = 0
    disfluency_count = 0
    prev_wrd = ""
    prev_dis = ""
    repeated_wrds = [str]
    repeated_counts = [int]
    code_switch_wrd = [str]
    for i in range(len_g):
        # first attempt to combine with adjacent word,
        # if nothing needs to be combined, should just return the current word
        gold_wrd = gold_pairs[i][0]
        gold_tag = gold_pairs[i][1]
        corrected_pair, offset = combine_adjacent(model_pairs, gold_pairs, i + total_offset, i)
        total_offset += offset
        model_wrd = corrected_pair[0].strip()
        model_tag = corrected_pair[1].strip()

        # offset == -1, that means the recombined pair does not match anything
        if offset == -1:
            raise MissingWordException(gold_wrd, model_wrd)
        # offset > 0, that means recombination is needed and a bad split occurred
        if offset > 0:
            bad_split += 1

        # if the word is the same, just compare tags
        if model_wrd == gold_wrd:
            if model_tag == gold_tag:
                correct += 1
            elif is_subclass(model_tag, gold_tag):
                correct += 1
            else:
                incorrect += 1
        # if model word is part of gold word, split is bad
        # check word bf and after and if combined is same, then check tag
        # else throw exception, missing word
        # TODO: behaviour might change when we compare hugginface, they split based on subwords
        else:
            # print(gold_standard)
            # print(model_generated)
            raise MissingWordException(gold_wrd, model_wrd)

        # handle counts of repeated words
        # TODO: also handle if repeated wrd is phrase, very unlikely
        # do not count if it is empty string
        if model_wrd != "":
            if model_wrd in repeated_wrds:
                index = repeated_wrds.index(model_wrd)
                repeated_counts[index] += 1
            elif model_wrd == prev_wrd:
                repeated_wrds.append(model_wrd)
                repeated_counts.append(1)
                repeated_count += 1
        prev_wrd = model_wrd

        # handle code switches
        if not gold_wrd.isascii():
            code_switch_count += 1
            code_switch_wrd.append(gold_wrd)

            # # find if it co-occurs with repeated or disfluencies
            # if repeated_count != 0:
            #     print(gold_wrd, (repeated_wrds[repeated_count - 1], repeated_counts[repeated_count - 1]))
            # if disfluency_count != 0:
            #     print(gold_wrd, prev_dis)

        # handle disfluencies
        if gold_tag in ["DIS", "FW", "UH"]:
            disfluency_count += 1
            prev_dis = gold_wrd

    # if code_switch_count > 0:
        # print(code_switch_wrd)
    return [correct, incorrect, bad_split, code_switch_count, repeated_count, disfluency_count]


def tagged_words_to_pairs(tagged_wrds: [str]):
    pairs = []
    for m_entry in tagged_wrds:
        entry = m_entry.split('(')
        if len(entry) == 1:
            e = entry[0].replace(" ", "")
            if e == "":
                continue
            else:
                raise MissingPOSTagException(e)
        else:
            word = entry[0]
            pos_tag = entry[1].replace(')', '')
            pairs.append((word, pos_tag))

    return pairs


def combine_adjacent(model_pairs, gold_pairs, model_pos: int, gold_pos: int):
    """takes in model_pairs and gold_pairs and the position that has the word that is weirdly tokenized
    eg. "Hong Kong" vs "Hong", "Kong", assume first word matches
    combine with adjacent words and also compare their tags if they are the same
    Could also be multiple adjacent words, eg. "Hong", "Pu" "Long" -> "Hong Pu Long"
    return the combined (word, tag) and offset
    offset is how many words are combined, this is for when we continue checking the words, we skip over the combined words
    iff the word does not need combining, just return the original word"""

    gold_match = gold_pairs[gold_pos]
    model_match = model_pairs[model_pos]
    model_substring = model_match[0]
    model_tag = model_match[1]
    offset = 0
    # stops when the word matches, or th
    for i in range(model_pos, len(model_pairs)):
        if model_substring == gold_match[0]:
            return (model_substring, model_tag), offset
        elif model_substring not in gold_match[0]:
            break
        # word is missing from
        elif i + 1 >= len(model_pairs):
            raise MissingWordException(gold_match[0], model_substring)

        # combine with next word and check if it matches gold
        model_substring = model_substring + ' ' + model_pairs[i + 1][0]
        # after combining, check if model tag is the same as previous tag
        if model_tag != model_pairs[i + 1][1]:
            model_tag = '-1'
        offset += 1

    return (model_substring, model_tag), -1


def combine_words(tokens, place, word, pos):
    substring = word
    offset = 0
    for i in range(pos, len(tokens)):
        # place name found
        if substring == place:
            break

        # cannot find place name
        if i >= len(tokens)-1 or substring not in place:
            substring = word
            offset = 0
            break

        # combine with next
        substring = substring + ' ' + tokens[i+1]
        offset += 1

    return substring, offset


def is_subclass(model_tag, gold_tag):
    adj = ["JJ", "JJR", "JJS"]
    noun = ["NN", "NNS", "NNP", "NNPS"]
    pronoun = ["PRP", "PRP$"]
    adv = ["RB", "RBR", "RBS"]
    verb = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    dis = ["DIS", "FW", "UH"]

    return ((model_tag in adj and gold_tag in adj) or
            (model_tag in noun and gold_tag in noun) or
            (model_tag in pronoun and gold_tag in pronoun) or
            (model_tag in adv and gold_tag in adv) or
            (model_tag in verb and gold_tag in verb) or
            (model_tag in dis and gold_tag in dis))


# TODO: if it is a code switch, how far before was the previous Dis? or Uh?
# TODO: evaluate tagging, does having more Dis or Uh or repeated words mess with tag accuracy?
def find_code_switch(model_pairs):
    code_switch_pos = []
    for i in range(len(model_pairs)):
        model_pair = model_pairs[i]
        if model_pair[0].strip() not in lower and model_pair[0].strip() not in upper:
            code_switch_pos.append(i)

    return code_switch_pos


def find_closest_csw(model_pairs, pos):
    closest_csw_pos = np.inf
    for i in range(len(model_pairs)):
        pair = model_pairs[i]
        if not pair[0].strip().isascii():
            pos_diff = abs(pos - i)
            closest_csw_pos = min(closest_csw_pos, pos_diff)

    return closest_csw_pos


def find_closest_dis(model_pairs, pos):
    closest_dis_pos = np.inf
    for i in range(len(model_pairs)):
        pair = model_pairs[i]
        if pair[1] in ["DIS", "FW", "UH"]:
            pos_diff = abs(pos - i)
            closest_dis_pos = min(closest_dis_pos, pos_diff)

    return closest_dis_pos


def find_closest_repeated(model_pairs, pos):
    closest_repeat_pos = np.inf
    prev_word = ""
    for i in range(len(model_pairs)):
        pair = model_pairs[i]
        if pair[0] == prev_word and pair[1] != "":
            pos_diff = abs(pos - i)
            closest_repeat_pos = min(closest_repeat_pos, pos_diff)
        prev_word = pair[0]

    return closest_repeat_pos


# TODO: the CRF for correction
# 2 layers: first for recombining words such as place names
#           second for correcting tagging
# features are the same
# input: sentences are pairs of (word, tag) from preliminary tagger
# output: sentences are pairs of (word, tag) with corrected tags
def word_features(tag_sentence, i):
    wrd_tag_pair = tag_sentence[i]
    word = wrd_tag_pair[0]
    tag = wrd_tag_pair[1]
    features = {
        'word': word,
        # 'tag': tag,
        'is_first': i == 0,
        'is_last': i == len(tag_sentence) - 1,
        # prefix of the word
        'prefix-1': "" if word == "" else word[0],
        'prefix-2': "" if word == "" else word[:2],
        'prefix-3': "" if word == "" else word[:3],
        # suffix of the word
        'suffix-1': "" if word == "" else word[-1],
        'suffix-2': "" if word == "" else word[-2:],
        'suffix-3': "" if word == "" else word[-3:],

        # custom features
        # extracting previous word
        'prev_word': '' if i == 0 else tag_sentence[i - 1][0],
        # extracting next word
        'next_word': '' if i == len(tag_sentence) - 1 else tag_sentence[i + 1][0],
        # disfluencies, code switches, repeated words
        'is_code_switch': not word.isascii(),
        'is_disfluency': word in ["mmm", "mnmm", "mmhm", "mm", "ahh", "huh", "umm", "mhm", "um"] or "&" in word,
        'is_label': "@" in word or "＠" in word,
        # placenames
        'is_place': word in PLACE_NAMES
    }

    return features


def preprocess(data):
    """recombine disfluency and label tags & and @
    recombine place names"""
    sym = ["&", "@", "＠"]

    recombined_sentences = []
    for sentence in data:
        tokens = nltk.word_tokenize(sentence)
        recombined_tokens = []
        total_offset = 0
        for i in range(len(tokens)):
            if i+total_offset >= len(tokens):
                break
            wrd = tokens[i+total_offset]
            if wrd == "":
                continue

            # recombine with next word if the word is part of sym
            if wrd in sym:
                wrd = wrd + tokens[i+total_offset+1]
                total_offset += 1

            # check if word is part of a place name, if yes, try and recombine
            for place in PLACE_NAMES:
                if wrd in place and len(wrd) > 1:
                    wrd, offset = combine_words(tokens, place, wrd, i+total_offset)
                    total_offset += offset

            recombined_tokens.append(wrd)
        recombined_sentences.append(recombined_tokens)
    return recombined_sentences

# sent = ["until i was i was born here in vancouver and i move to hong kong when i was three cuz my parents are from hong kong "]
# recom = preprocess(sent)
# print(recom)

def preliminary_tag(preprocess):
    """Take each sentence in corpus and tag them
    RETURN: corpus of tagged sentences"""
    tagged_corpus = []
    for sentence in preprocess:
        tagged_sentence = nltk.pos_tag(sentence)
        tagged_corpus.append(tagged_sentence)

    return tagged_corpus


def correction_crf(tagged_corpus):
    X = []
    y = []
    for sentence in tagged_corpus:
        X_sentence = []
        y_sentence = []
        for i in range(len(sentence)):
            X_sentence.append(word_features(sentence, i))
            y_sentence.append(sentence[i][1])
        X.append(X_sentence)
        y.append(y_sentence)

    # Train a CRF model on the training data
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=10000,
        all_possible_transitions=True
    )
    crf.fit(X, y)

    # Make predictions on the test data and evaluate the performance
    # y_pred = crf.predict(X_test)
    #
    # print(metrics.flat_accuracy_score(y_test, y_pred))
    return crf


# Exception classes
class MissingWordException(Exception):
    def __init__(self, gold_wrd, model_wrd):
        self.gold_wrd = gold_wrd
        self.model_wrd = model_wrd

    def __str__(self):
        return repr(f'Missmatch Word: should be {self.gold_wrd}, instead {self.model_wrd}')


class UnexpectedTokenizationException(Exception):
    def __init__(self, gold_split, model_split):
        self.gold_split = gold_split
        self.model_split = model_split

    # TODO: fix the new lines to the message
    def __str__(self):
        return repr(f'Unexpected tokenization difference: hand corrected version tokenized more. '
                    f'gold: {self.gold_split}. model: {self.model_split}')


class MissingPOSTagException(Exception):
    def __init__(self, word):
        self.word = word

    def __str__(self):
        return repr(f'{self.word} is missing a POS tag')



# TODO: what are the pos that needs the most correcting
# model = rd.ParsedTextgrid('Prat data/tagged_interviews_textgrid/POS_VM24A_English_I1_20181209.TextGrid')
# gold = rd.ParsedTextgrid('Prat data/corrected_tagged_textgrid/POS_VM24A_DT_Correct_English_I1_20181209_DT_Edited.TextGrid')
# model_generated = model.get_tiers()[2].get_intervals()[3].get_text()
# gold_standard = gold.get_tiers()[2].get_intervals()[3].get_text()
# print(model_generated, gold_standard)
# find_text_diff(model_generated, gold_standard)
# # count if the model ever have longer words than gold standard
# model_IU = model.get_tiers()[2].get_intervals()
# gold_IU = gold.get_tiers()[2].get_intervals()
# gold_more = 0
# for t in range(len(model_IU)):
#     gold_more += find_text_diff(model_IU[t].get_text(), gold_IU[t].get_text())
# print(gold_more)


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
