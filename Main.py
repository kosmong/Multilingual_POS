from fontTools.misc import filenames

import Read_Data as Read
import Write_Data as Write
import Multilingual_POS as MPOS
import pandas as pd
import nltk
import sklearn_crfsuite
from sklearn_crfsuite import metrics

CSV_FOLDER = "Prat data/tagged_interviews_csv/"
TEXTGRID_FOLDER = "Prat data/interview_textgrids_iu_and_cs_intervals/"
TAGGED_FOLDER = "Prat data/tagged_interviews_textgrid/"
HUGGINGFACE_FOLDER = "Prat data/tagged_huggingface_textgrid/"
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

corrected_files = ["Prat data/corrected_tagged_textgrid/POS_VM19A_DT_English_I1_20191031.TextGrid",
                   "Prat data/corrected_tagged_textgrid/POS_VM19B_DT_English_I2_20191104.TextGrid",
                   "Prat data/corrected_tagged_textgrid/POS_VM24A_DT_Correct_English_I1_20181209_DT_Edited.TextGrid",
                   "Prat data/corrected_tagged_textgrid/Regan_POS_VM24A_English_I1_20181209.TextGrid",
                   "Prat data/corrected_tagged_textgrid/SW_POS_VF19D_English_I2_20190308.TextGrid"]
model_files = ["Prat data/interview_textgrids_iu_and_cs_intervals/VM19A_English_I1_20191031.TextGrid",
               "Prat data/interview_textgrids_iu_and_cs_intervals/VM19B_English_I2_20191104.TextGrid",
               "Prat data/interview_textgrids_iu_and_cs_intervals/VM24A_English_I1_20181209.TextGrid",
               "Prat data/interview_textgrids_iu_and_cs_intervals/VM24A_English_I1_20181209.TextGrid",
               "Prat data/interview_textgrids_iu_and_cs_intervals/VF19D_English_I2_20190308.TextGrid"]

min = 0


# print("xmin = " + str(min) + " ")


# parsed = Read.ParsedTextgrid("Prat data/interview_textgrids_iu_and_cs_intervals/VF19A_English_I1_20181114.TextGrid")
# write_data(parsed, filename="Prat data/tagged_interviews_textgrid/POS_VM24A_DT_Correct_English_I1_20181209_DT_Edited.TextGrid")


def main():
    # # read textgrid, add preliminary POS tagging into the convienience-UI tier, write textgrid
    # for filename in FILES_NAMES:
    #     old_parsed = Read.ParsedTextgrid(TEXTGRID_FOLDER + filename + ".TextGrid")
    #     # Write.write_data(old_parsed, HUGGINGFACE_FOLDER + "POS_hf_" + filename + ".TextGrid", 'huggingface')
    #     Write.write_data(old_parsed, TAGGED_FOLDER + "POS_" + filename + ".TextGrid")

    for i in range(len(model_files)):
        get_correction_counts(model_files[i], corrected_files[i])
    # total_counts()

    data = []
    for file in FILES_NAMES:
        filename = (TEXTGRID_FOLDER + file + ".TextGrid")
        textgrid = Read.ParsedTextgrid(filename)
        cUI = textgrid.get_tiers()[2].get_intervals()
        for j in range(len(cUI)):
            sentence = cUI[j].get_text()
            data.append(sentence)
    crf = pipeline_train(data)

    X = []
    y = []
    test_corp = []
    for filename in corrected_files:
        gold_grid = Read.ParsedTextgrid(filename)
        cUI_intervals = gold_grid.get_tiers()[2].get_intervals()
        for j in range(len(cUI_intervals)):
            pairs = MPOS.tagged_words_to_pairs(cUI_intervals[j].get_text().split(') '))
            if not (len(pairs) == 1 and pairs[0][0] == "" and pairs[0][1] == ""):
                test_corp.append(pairs)

    for sentence in test_corp:
        X_sentence = []
        y_sentence = []
        for i in range(len(sentence)):
            X_sentence.append(MPOS.word_features(sentence, i))
            y_sentence.append(sentence[i][1])
        X.append(X_sentence)
        y.append(y_sentence)

    split = int(0.8 * len(X))
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    # Train a CRF model on the training data
    crf_gold = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=10000,
        all_possible_transitions=True
    )
    crf_gold.fit(X_train, y_train)
    y_pred_prelim = crf.predict(X_test)
    y_pred_gold = crf_gold.predict(X_test)
    print(metrics.flat_accuracy_score(y_test, y_pred_prelim))
    print(metrics.flat_accuracy_score(y_test, y_pred_gold))




def get_correction_counts(model, gold):
    # Compare the tagged textgrid of nltk to gold standard, print counts
    model_grid = Read.ParsedTextgrid(model)
    gold_grid = Read.ParsedTextgrid(gold)

    model_cUI_intervals = model_grid.get_tiers()[2].get_intervals()
    gold_cUI_intervals = gold_grid.get_tiers()[2].get_intervals()
    assert len(model_cUI_intervals) == len(gold_cUI_intervals)
    model_data = []
    gold_data = []
    for j in range(len(model_cUI_intervals)):
        sentence_m = model_cUI_intervals[j].get_text()
        model_data.append(sentence_m)
        sentence_g = MPOS.tagged_words_to_pairs(gold_cUI_intervals[j].get_text().split(') '))
        gold_data.append(sentence_g)

    pre_m = MPOS.preprocess(model_data)
    tagged_m = MPOS.preliminary_tag(pre_m)
    tagged_g = gold_data

    correct_count = 0
    incorrect_count = 0
    bad_split_count = 0
    csw_count = 0
    repeated_count = 0
    disfluencies = 0
    correction_index = []
    for i in range(len(model_cUI_intervals)):
        model_text = tagged_m[i]
        gold_text = tagged_g[i]

        # print(i)
        result = MPOS.find_text_diff(model_text, gold_text)
        correct = result[0]
        incorrect = result[1]
        bad_split = result[2]
        csw = result[3]
        repeated = result[4]
        disfluency = result[5]
        if incorrect > 0:
            correction_index.append(i + 1)
        correct_count += correct
        incorrect_count += incorrect
        bad_split_count += bad_split
        csw_count += csw
        repeated_count += repeated
        disfluencies += disfluency

    print(f"Correct = {correct_count}\n"
          f"Incorrect = {incorrect_count}\n"
          f"Bad split = {bad_split_count}\n"
          f"Code switch count = {csw_count}\n"
          f"Repeated = {repeated_count}\n"
          f"Disfluency = {disfluencies}\n"
          f"Correct Percent = {correct_count / (correct_count + incorrect_count)}\n")
          # f"Corrections in: {correction_index}")


def total_counts():
    code_switch_count = 0
    disfluency_count = 0
    repeated_count = 0
    total_word = 0
    co_occurrences = {"csw_csw": 0, "csw_dis": 0, "csw_rep": 0}
    continuous_csw = []
    csw_tags = {}
    for filename in FILES_NAMES:
        grid = Read.ParsedTextgrid(TAGGED_FOLDER + "POS_" + filename + ".TextGrid")

        cUI_intervals = grid.get_tiers()[2].get_intervals()
        for j in range(len(cUI_intervals)):
            pairs = MPOS.tagged_words_to_pairs(cUI_intervals[j].get_text().split(') '))
            prev = None
            cont_csw = 0
            had_csw = False
            had_dis = False
            had_rep = False
            for pair in pairs:
                wrd = pair[0]
                tag = pair[1]

                # handle code switches
                if not wrd.isascii():
                    code_switch_count += 1
                    had_csw = True
                    if had_csw:
                        co_occurrences["csw_csw"] += 1
                    if had_dis:
                        co_occurrences["csw_dis"] += 1
                    if had_rep:
                        co_occurrences["csw_rep"] += 1
                    if prev is not None and not prev[0].isascii():
                        cont_csw += 1
                    if tag in csw_tags:
                        csw_tags[tag] += 1
                    else:
                        csw_tags.update({tag: 1})


                # handle disfluencies
                if tag in ["DIS", "FW", "UH"]:
                    disfluency_count += 1
                    had_dis = True

                # handle repeated
                if prev is not None and wrd == prev[0]:
                    repeated_count += 1
                    had_rep = True

                prev = pair
                total_word += 1
            if cont_csw > 5:
                continuous_csw.append(pairs)

    print(f"Code switch = {code_switch_count}\n"
          f"Disfluency = {disfluency_count}\n"
          f"Repeated = {repeated_count}\n"
          f"Csw with Csw = {co_occurrences["csw_csw"]}\n"
          f"Csw with Dis = {co_occurrences["csw_dis"]}\n"
          f"Csw with Rep = {co_occurrences["csw_rep"]}\n"
          f"Continuous csw = {continuous_csw}\n"
          f"Csw tags = {csw_tags}\n"
          f"Total = {total_word}")


def try_model():
    # try the model
    # first, get corpus: hand tagged sentences
    corpus = []
    for filename in FILES_NAMES:
        filename = TAGGED_FOLDER + "POS_" + filename + ".TextGrid"
        gold_grid = Read.ParsedTextgrid(filename)
        cUI_intervals = gold_grid.get_tiers()[2].get_intervals()
        for j in range(len(cUI_intervals)):
            pairs = MPOS.tagged_words_to_pairs(cUI_intervals[j].get_text().split(') '))
            if not (len(pairs) == 1 and pairs[0][0] == "" and pairs[0][1] == ""):
                corpus.append(pairs)

    X = []
    y = []
    for sentence in corpus:
        X_sentence = []
        y_sentence = []
        for i in range(len(sentence)):
            X_sentence.append(MPOS.word_features(sentence, i))
            y_sentence.append(sentence[i][1])
        X.append(X_sentence)
        y.append(y_sentence)

    # print(len(X), len(y))
    # print(len(X[0]), len(y[0]))
    # print(X[0], y[0])

    # split = int(0.8 * len(X))
    # X_train = X[:split]
    # y_train = y[:split]
    # X_test = X[split:]
    # y_test = y[split:]
    X_train = X
    y_train = y

    X_test = []
    y_test = []
    test_corp = []
    for filename in corrected_files:
        gold_grid = Read.ParsedTextgrid(filename)
        cUI_intervals = gold_grid.get_tiers()[2].get_intervals()
        for j in range(len(cUI_intervals)):
            pairs = MPOS.tagged_words_to_pairs(cUI_intervals[j].get_text().split(') '))
            if not (len(pairs) == 1 and pairs[0][0] == "" and pairs[0][1] == ""):
                test_corp.append(pairs)

    for sentence in test_corp:
        X_sentence = []
        y_sentence = []
        for i in range(len(sentence)):
            X_sentence.append(MPOS.word_features(sentence, i))
            y_sentence.append(sentence[i][1])
        X_test.append(X_sentence)
        y_test.append(y_sentence)

    # Train a CRF model on the training data
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=10000,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    # Make predictions on the test data and evaluate the performance
    y_pred = crf.predict(X_test)

    print(metrics.flat_accuracy_score(y_test, y_pred))


# pipeline_train takes in the data as a list of sentences
# it preprocesses the sentences, preliminarily tag it, then train a crf using the tagged sentences
# returns the trained crf
def pipeline_train(data):
    preprocessed = MPOS.preprocess(data)
    corpus = MPOS.preliminary_tag(preprocessed)
    crf = MPOS.correction_crf(corpus)
    return crf

if __name__ == "__main__":
    main()

# test, _ = POS_sentence("i like apples")
# sym = ["&", "@", "＠"]
# disf = ["&", "mmm", "mnmm", "mmhm", "mm", "ahh", "huh", "umm"]
# test_tag = recombine_and_retag(test, sym, disf)
# print(test)
#
# one_recombine = "i &uh like apples"
# tagged_one_recombine, _ = POS_sentence(one_recombine)
# print(tagged_one_recombine)
# retag_one_recombine = recombine_and_retag(tagged_one_recombine, sym, disf)
# print(retag_one_recombine)
# print(tagged_one_recombine)
# print('&' == tagged_one_recombine[1][0])
#
# consecutive_recombine = "i &uh @um @hum like apples"
# tagged_consecutive_recombine, _ = POS_sentence(consecutive_recombine)
# retag_consecutive_recombine = recombine_and_retag(tagged_consecutive_recombine, sym, disf)
# print(retag_consecutive_recombine)
# print(tagged_consecutive_recombine)

# # parse textgrid, store as dataframe, save as csv file
# for filename in FILES_NAMES:
#     path = TEXTGRID_FOLDER + filename + ".TextGrid"
#     parsed_textgrid = rd.ParsedTextgrid(path)
#     convenience_tier = parsed_textgrid.get_tiers()[2]
#     assert convenience_tier.get_name() == "convenience-IU"
#     textgrid_into_csv(filename, convenience_tier)


# make master count df
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
# example_canto = "楊枝甘露 is delicious"
# tagged_ex, _ = POS_sentence(example_canto)
# print(tagged_ex)
#
# # I think this is correct
# tagged_canto2, _ = POS_sentence("my favourite 湯圓 is still 黑芝麻")
# tagged_eng, _ = POS_sentence("my favourite tang yuan is still sesame")
# print(tagged_canto2)
# print(tagged_eng)


# TODO: NOTES
# # get list of code switched words and number of counts of each 0-408 in master counts
# mcount_df = pd.read_csv(CSV_FOLDER + 'Master_Counts.csv', index_col=0)
# mcount_df = mcount_df[1:407]
# print(mcount_df)
# test = pd.DataFrame({'Words':['a', 'b'], "Count":[[('NP',2)],[('NP',3), ('O',3)]]})
# print(test)
# csw_count_df = make_csw_count(mcount_df)
# print(csw_count_df)

# & means discontinuations, must stay with the next word (done)
# check unique words and maybe hand tag
# discontinuations and disfluencies: false start (&) um, ahhs, mmm
# cantonese testing, where is the pos different?
# @ sign for code-switching: language other than english or canto
# @m: mandarin, @j: japanese, @ml: malay and @i: indonesian; 21a, ＠ in 21c, 22a, 24a, 25a, 27a
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

# crf word order, feature vectors
# we can lengthen as well
# how much labelled data?
# special cases?: disfluencies, corrections, deletions
# HMM or CRFs
# direct tl

# send molly preliminary tagged

# TODO:
# how to combine different tokenized pos
# make a new multilevel conditional random field
# with variable length features that detect, word, phrasal (NP, VP), sentence
