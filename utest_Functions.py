import unittest
import Read_Data as Read
import numpy as np
import pandas as pd
import Multilingual_POS as MPOS

PATH = "Prat data/interview_textgrids_iu_and_cs_intervals/"
FILES = ["VF19A_English_I1_20181114.TextGrid",
         "VF19B_English_I1_20190213.TextGrid",
         "VF19C_English_I1_20190224.TextGrid",
         "VF19D_English_I2_20190308.TextGrid",
         "VF20A_English_I2_20181119.TextGrid",
         "VF20B_English_I2_20181203.TextGrid",
         "VF21A_English_I1_20190130.TextGrid",
         "VF21B_English_I2_20190204.TextGrid",
         "VF21C_English_I2_20190211.TextGrid",
         "VF21D_English_I1_20190306.TextGrid",
         "VF22A_English_I2_20181206.TextGrid",
         "VF23B_English_I1_20190121.TextGrid",
         "VF23C_English_I2_20190128.TextGrid",
         "VF26A_English_I2_20190303.TextGrid",
         "VF27A_English_I1_20181120.TextGrid",
         "VF32A_English_I2_20190213.TextGrid",
         "VF33B_English_I1_20190206.TextGrid",
         "VM19A_English_I1_20191031.TextGrid",
         "VM19B_English_I2_20191104.TextGrid",
         "VM19C_English_I1_20200211.TextGrid",
         "VM19D_English_I2_20200211.TextGrid",
         "VM20B_English_I1_20181126.TextGrid",
         "VM21A_English_I1_20181206.TextGrid",
         "VM21B_English_I1_20190313.TextGrid",
         "VM21C_English_I2_20190403.TextGrid",
         "VM21D_English_I2_20200309.TextGrid",
         "VM21E_English_I2_20200309.TextGrid",
         "VM22A_English_I2_20181210.TextGrid",
         "VM22B_English_I1_20200309.TextGrid",
         "VM23A_English_I1_20200310.TextGrid",
         "VM24A_English_I1_20181209.TextGrid",
         "VM25A_English_I1_20190923.TextGrid",
         "VM25B_English_I1_20200224.TextGrid",
         "VM34A_English_I2_20191028.TextGrid"]


class TestMPOS(unittest.TestCase):
    def setUp(self) -> None:
        self.filenames = []
        for file in FILES:
            self.filenames.append(PATH + file)

    # def test_tier_interval_nums(self):
    #     # test
    #     for filename in self.filenames:
    #         parsed_ts = rd.ParsedTextgrid(filename)
    #         tier_nums = parsed_ts.get_tier_num()
    #         tiers = parsed_ts.get_tiers()
    #         self.assertEqual(tier_nums, len(tiers))
    #
    #         for tier in tiers:
    #             interval_nums = tier.get_interval_num()
    #             intervals = tier.get_intervals()
    #             self.assertEqual(interval_nums, len(intervals))

    def test_POS_sentence(self):
        # test POS tagger
        #
        # # test out how nltk work
        # simple_eng = "I went to the mall."
        # tagged_eng, cleaned_eng = POS_sentence(simple_eng)
        #
        # # test for canto
        # simple_canto = "古箏"
        # tagged_simple, cleaned_simple = POS_sentence(simple_canto)
        #
        # # test for code switched sentence
        # csw_sen = 'i like learned 古箏 for like a while'
        # tagged_csw, cleaned_csw = POS_sentence(csw_sen)
        # csw_wrds, csw_tags = zip(*tagged_csw)
        # csw_df = pd.DataFrame({'POS tags': csw_tags, 'Word tokens': csw_wrds})
        #
        # csw_sen2 = 'they had this 古箏 teacher that volunteered to like teach us'
        # tagged_csw2, cleaned_csw2 = POS_sentence(csw_sen2)
        # csw_a2, csw_b2 = zip(*tagged_csw2)
        # csw2_df = pd.DataFrame({'POS tags': csw_a2, 'Word tokens': csw_b2})
        # df = pd.concat([csw_df, csw2_df])
        #
        # longer_sen = ("mmm oh i also learned the 古箏 like yeah. "
        #               "i don't know like i was like just putting it out there like just cuz we were on the topic and i was like i totally forgot yeah. "
        #               "cuz at like temple they had like 古箏 like they had this 古箏 teacher that volunteered to like teach us so i learned 古箏 for a while i have 古箏 at home and like. "
        #               "yeah it's like it's pretty fun actually um uh and any instrument that i want to play okay if i ever wanted to try something i'd probably try like the tuba or something. "
        #               "just cuz like everything i've played was like strings related and stuff but i've never tried like like a wind instrument. "
        #               "but i know also like controlling my breath is probably pretty hard cuz i don't know i know you have to have like the um embrasure or something like that. "
        #               "for any like like those weird instruments but a tuba would be cool")
        # tagged_long, cleaned_long = POS_sentence(longer_sen)
        # a, b = zip(*tagged_long)

        # big_test_df = pd.read_csv('Prat data/tagged_interviews_csv/VF19A_English_I1_20181114.csv')
        # big_count = make_count_df(big_test_df)
        # print(big_count)

        # test_data = {'POS tags': ['N', 'N', 'V'], 'Words': ['apple', 'orange', 'eat'], 'Count': [1, 2, 3]}
        # test_df = pd.DataFrame(data=test_data)
        # print(test_df)
        # # test logic of both pos and word matching
        # f_i = test_df.loc[(test_df['POS tags'] == 'N') & (test_df['Words'] == 'apple')].index
        # print(f_i)
        # print(test_df.loc[f_i])
        # test_df.loc[f_i, 'Count'] = test_df.loc[f_i, 'Count'] + 1
        # print(test_df)
        # fail_row = test_df.loc[(test_df['POS tags'] == 'V') & (test_df['Words'] == 'run')].index
        # print(fail_row.empty)

        # example_df = pd.DataFrame(data={'POS tags': ['N', 'N', 'V', 'N'], 'Words': ['apple', 'orange', 'eat', 'apple']})
        # print(example_df)
        # columns = ['POS tags', 'Words', 'Count']
        # count_df = pd.DataFrame(columns=columns)
        # example_count = make_count_df(example_df, count_df)
        # print(example_count)
        # example_count.sort_values('Words', inplace=True, ascending=False)
        # print(example_count)

        # m_count_df = pd.read_csv(CSV_FOLDER + 'Master_Counts.csv', index_col=0)
        # m_count_df.sort_values(by='Words', ascending=False)
        # m_count_df.to_csv(CSV_FOLDER + 'Master_Counts.csv')

        assert True

    # This tests the the recombine_and_retag function in Multilingual_POS
    # Recombines [&, @] with next element
    # cases: empty, one, ten, many
    # also test retagging disfluencies
    # cases: empty, one, ten, many
    # test them together
    def test_recombine_and_retag(self):
        # recombine
        sym = ["&", "@", "＠"]
        disf = ["&", "mmm", "mnmm", "mmhm", "mm", "ahh", "huh", "umm"]

        no_recombine = "i like apples"
        tagged_no_recombine, _ = MPOS.POS_sentence(no_recombine)
        retag_no_recombine = MPOS.recombine_and_retag(tagged_no_recombine, sym, disf)
        # before is the same as after, should have no changes
        self.assertEqual(tagged_no_recombine, retag_no_recombine)

        one_recombine = "i &uh like apples"
        tagged_one_recombine, _ = MPOS.POS_sentence(one_recombine)
        retag_one_recombine = MPOS.recombine_and_retag(tagged_one_recombine, sym, disf)
        self.assertEqual(len(retag_one_recombine), len(tagged_one_recombine) - 1)
        # elements in the sym list should not occur by itself
        for s in sym:
            for t in retag_one_recombine:
                self.assertFalse(s == t[0])
        correct_retag_one = [('i', 'NN'), ('&uh', 'DIS'), ('like', 'IN'), ('apples', 'NNS')]
        self.assertEqual(correct_retag_one, retag_one_recombine)

        consecutive_recombine = "i &uh @um @hum like apples"
        tagged_consecutive_recombine, _ = MPOS.POS_sentence(consecutive_recombine)
        retag_consecutive_recombine = MPOS.recombine_and_retag(tagged_consecutive_recombine, sym, disf)
        self.assertEqual(len(retag_consecutive_recombine), len(tagged_consecutive_recombine) - 3)
        for s in sym:
            for t in retag_one_recombine:
                self.assertFalse(s == t[0])
        correct_retag_consecutive = [('i', 'NN'), ('&uh', 'DIS'), ('@um', 'LABEL'), ('@hum', 'LABEL'), ('like', 'IN'),
                                     ('apples', 'NNS')]
        self.assertEqual(correct_retag_consecutive, retag_consecutive_recombine)

        one_disfluency = "I um like apples"
        tagged_one_disfluency, _ = MPOS.POS_sentence(one_disfluency)
        retag_one_disfluency = MPOS.recombine_and_retag(tagged_one_disfluency, sym, disf)
        self.assertEqual(len(retag_one_disfluency), len(tagged_one_disfluency))

    def test_find_text_diff(self):
        """Test the combine_and_retag function
        case 1: gold > model, exception raised, usually gold standard should have shorter lengths if they are different
        case 2a: gold < model and recombine matches, output should gold = model, model is recombined and it matches words of gold
        case 2b: gold < model but recombine does not match, exception raised, words are different which should not occur
        case 3: gold = model and there is no need for recombination"""

        # case1: gold > model, exception raised, hand corrected has more tokenizing
        case1_model_sentence = "uh(DIS) i(NN) 'm(VBP) from(IN) hong kong(NNP) um(DIS) i(PRP) well(CC)"
        case1_gold_sentence = "uh(NN) i(NN) 'm(VBP) from(IN) hong(NN) kong(NN) um(DIS) i(RB) well(RB)"
        with self.assertRaises(MPOS.UnexpectedTokenizationException):
            MPOS.find_text_diff(case1_model_sentence, case1_gold_sentence)
        # MPOS.find_text_diff(case1_model_sentence, case1_gold_sentence)

        # case2a: gold < model, need recombination, successful
        case2a_model_sentence = "uh(NN) i(NN) 'm(VBP) from(IN) hong(NN) kong(NN) la(NNP) um(DIS) i(RB) well(RB)"
        case2a_gold_sentence = "uh(DIS) i(NN) 'm(VBP) from(IN) hong kong la(NNP) um(DIS) i(PRP) well(CC)"
        result2a = MPOS.find_text_diff(case2a_model_sentence, case2a_gold_sentence)
        correct2a = result2a[0]
        incorrect2a = result2a[1]
        bad_split2a = result2a[2]
        assert correct2a == 4
        assert incorrect2a == 4
        assert bad_split2a == 1

        # case2amany: many successful
        case2amany_model_sentence = "uh(NN) i(NN) 'm(VBP) from(IN) hong(NN) kong(NN) la(NNP) um(DIS) i(RB) well(RB) long(JJ) ma(JJ) zhi(JJ)"
        case2amany_gold_sentence = "uh(DIS) i(NN) 'm(VBP) from(IN) hong kong la(NNP) um(DIS) i(PRP) well(CC) long ma zhi(JJ)"
        result2amany = MPOS.find_text_diff(case2amany_model_sentence, case2amany_gold_sentence)
        correct2amany = result2amany[0]
        incorrect2amany = result2amany[1]
        bad_split2amany = result2amany[2]
        print(correct2amany, incorrect2amany, bad_split2amany)
        assert correct2amany == 5
        assert incorrect2amany == 4
        assert bad_split2amany == 2

        # case2b: gold < model, recombination finds word mismatch
        case2b_model_sentence = "uh(NN) i(NN) 'm(VBP) from(IN) hong(NN) pong(NN) la(NNP) um(DIS) i(RB) well(RB)"
        case2b_gold_sentence = "uh(DIS) i(NN) 'm(VBP) from(IN) hong kong la(NNP) um(DIS) i(PRP) well(CC)"
        with self.assertRaises(MPOS.MissingWordException) as miss_wrd:
            MPOS.find_text_diff(case2b_model_sentence, case2b_gold_sentence)
        exception2b = miss_wrd.exception
        print(exception2b.model_wrd)
        print(exception2b.gold_wrd)
        assert exception2b.model_wrd == "hong pong"

        # case2bmany: many but one does not work, so should fail at the end
        case2bmany_model_sentence = "uh(NN) i(NN) 'm(VBP) from(IN) hong(NN) kong(NN) la(NNP) um(DIS) i(RB) well(RB) bing(NN) num(NN)"
        case2bmany_gold_sentence = "uh(DIS) i(NN) 'm(VBP) from(IN) hong kong la(NNP) um(DIS) i(PRP) well(CC) bing lum(NN)"
        with self.assertRaises(MPOS.MissingWordException) as miss_wrd:
            MPOS.find_text_diff(case2bmany_model_sentence, case2bmany_gold_sentence)
        exception2bmany = miss_wrd.exception
        assert "hong kong la" != exception2bmany.model_wrd
        print(exception2bmany.model_wrd)
        print(exception2bmany.gold_wrd)
        assert "bing num" == exception2bmany.model_wrd

        # case2c: missing a word to combine, exception thrown
        case2c_model_sentence = "uh(NN) i(NN) 'm(VBP) from(IN) hong(NN) kong(NN) la(NNP) um(DIS) i(RB) well(RB) ling(NN)"
        case2c_gold_sentence = "uh(DIS) i(NN) 'm(VBP) from(IN) hong kong la(NNP) um(DIS) i(PRP) well(CC) bing lum(NN)"
        with self.assertRaises(MPOS.MissingWordException) as miss_wrd:
            MPOS.find_text_diff(case2c_model_sentence, case2c_gold_sentence)
        exceptionc = miss_wrd.exception
        print(exceptionc.model_wrd)
        print(exceptionc.gold_wrd)
        assert exceptionc.model_wrd == "ling"

        # case3: no need for recombination
        case3_model_sentence = "uh(DIS) i(NN) 'm(VBP) from(IN) hong kong la(NNP) um(DIS) i(PRP) well(CC) bing lum(NN)"
        case3_gold_sentence = "uh(DIS) i(NN) 'm(VBP) from(IN) hong kong la(NNP) um(DIS) i(PRP) well(CC) bing lum(NN)"
        result3 = MPOS.find_text_diff(case3_model_sentence, case3_gold_sentence)
        correct3 = result3[0]
        incorrect3 = result3[1]
        bad_split3 = result3[2]
        assert correct3 == 9
        assert incorrect3 == 0
        assert bad_split3 == 0

        # case4: recombination success and tags are a subset eg. NN and NNP
        case4_model_sentence = "uh(NN) i(NN) 'm(VBP) from(IN) hong(NN) kong(NN) la(NN) um(DIS) i(RB) well(RB)"
        case4_gold_sentence = "uh(DIS) i(NN) 'm(VBP) from(IN) hong kong la(NNP) um(DIS) i(PRP) well(CC)"
        result4 = MPOS.find_text_diff(case4_model_sentence, case4_gold_sentence)
        correct4 = result4[0]
        incorrect4 = result4[1]
        bad_split4 = result4[2]
        assert correct4 == 5
        assert incorrect4 == 3
        assert bad_split4 == 1

        # case4many: many successful
        case4many_model_sentence = "uh(NN) i(NN) 'm(VBP) from(IN) hong(NN) kong(NN) la(NNP) um(DIS) i(RB) well(RB) long(JJ) ma(JJ) zhi(JJ)"
        case4many_gold_sentence = "uh(DIS) i(NN) 'm(VBP) from(IN) hong kong la(NNP) um(DIS) i(PRP) well(CC) long ma zhi(JJR)"
        result4many = MPOS.find_text_diff(case4many_model_sentence, case4many_gold_sentence)
        correct4many = result4many[0]
        incorrect4many = result4many[1]
        bad_split4many = result4many[2]
        assert correct4many == 5
        assert incorrect4many == 4
        assert bad_split4many == 2

    def test_combine_adjacent(self):
        """test combine_adjacent function
        case 1: cannot recombine adjacent words, they do not match the word of gold standard
        case 2: no need to recombine adjacent words, just return original word
        case 3: recombination is a success! also tags matches, only 1 tag, also include multiple cont
        case 4: recombination success but tags does not match, many tags
        case 5: missing word, we reached the end of  pairs, throuw missingwordexception"""

        # case1: cannot find correct recombine
        case1_mpairs = [("i", "NN"), ("was", "VBD"), ("born", "VBN"), ("in", "IN"), ("hong", "NN"), ("pong", "NN")]
        case1_gpairs = [("i", "NN"), ("was", "VBD"), ("born", "VBN"), ("in", "IN"), ("hong kong", "NNP")]
        case1_pos = 4
        result1_pair, offset1 = MPOS.combine_adjacent(case1_mpairs, case1_gpairs, case1_pos, case1_pos)
        print(result1_pair, offset1)
        assert offset1 == -1

        # case 2: no need to recombine
        case2_mpairs = [("i", "NN"), ("was", "VBD"), ("born", "VBN"), ("in", "IN"), ("hong", "NN"), ("kong", "NN")]
        case2_gpairs = [("i", "NN"), ("was", "VBD"), ("born", "VBN"), ("in", "IN"), ("hong", "NN"), ("kong", "NN")]
        case2_pos = 4
        result2_pair, offset2 = MPOS.combine_adjacent(case2_mpairs, case2_gpairs, case2_pos, case2_pos)
        print(result2_pair, offset2)
        assert offset2 == 0
        assert case2_gpairs[case2_pos] == result2_pair

        # case3: successful recombination, correct tag
        case3a_mpairs = [("i", "NN"), ("was", "VBD"), ("born", "VBN"), ("in", "IN"), ("hong", "NN"), ("kong", "NN")]
        case3a_gpairs = [("i", "NN"), ("was", "VBD"), ("born", "VBN"), ("in", "IN"), ("hong kong", "NN")]
        case3a_pos = 4
        result3a_pair, offset3a = MPOS.combine_adjacent(case3a_mpairs, case3a_gpairs, case3a_pos, case3a_pos)
        print(result3a_pair, offset3a)
        assert case3a_gpairs[case3a_pos] == result3a_pair
        assert offset3a == 1

        case3b_mpairs = [("i", "NN"), ("was", "VBD"), ("born", "VBN"), ("in", "IN"), ("hong", "NN"), ("kong", "NN"),
                         ("la", "NN")]
        case3b_gpairs = [("i", "NN"), ("was", "VBD"), ("born", "VBN"), ("in", "IN"), ("hong kong la", "NN")]
        case3b_pos = 4
        result3b_pair, offset3b = MPOS.combine_adjacent(case3b_mpairs, case3b_gpairs, case3b_pos, case3b_pos)
        print(result3b_pair, offset3b)
        assert case3b_gpairs[case3b_pos] == result3b_pair
        assert offset3b == 2

        # successful recombinations, incorrect tags
        case4a_mpairs = [("i", "NN"), ("was", "VBD"), ("born", "VBN"), ("in", "IN"), ("hong", "JJ"), ("kong", "JJ")]
        case4a_gpairs = [("i", "NN"), ("was", "VBD"), ("born", "VBN"), ("in", "IN"), ("hong kong", "NNP")]
        case4a_pos = 4
        result4a_pair, offset4a = MPOS.combine_adjacent(case4a_mpairs, case4a_gpairs, case4a_pos, case4a_pos)
        print(result4a_pair, offset4a)
        assert case4a_gpairs[case4a_pos][0] == result4a_pair[0]
        assert case4a_gpairs[case4a_pos][1] != result4a_pair[1]
        assert result4a_pair[1] == "JJ"
        assert not MPOS.is_subclass(result4a_pair[1], case4a_gpairs[case4a_pos][1])
        assert offset4a == 1

        case4b_mpairs = [("i", "NN"), ("was", "VBD"), ("born", "VBN"), ("in", "IN"), ("hong", "NNP"), ("kong", "NN"),
                         ("la", "NN")]
        case4b_gpairs = [("i", "NN"), ("was", "VBD"), ("born", "VBN"), ("in", "IN"), ("hong kong la", "NNP")]
        case4b_pos = 4
        result4b_pair, offset4b = MPOS.combine_adjacent(case4b_mpairs, case4b_gpairs, case4b_pos, case4b_pos)
        print(result4b_pair, offset4b)
        assert case4b_gpairs[case4b_pos][0] == result4b_pair[0]
        assert case4b_gpairs[case4b_pos][1] != result4b_pair[1]
        assert result4b_pair[1] == "-1"
        assert not MPOS.is_subclass(result4b_pair[1], case4b_gpairs[case4b_pos][1])
        assert offset4b == 2

        # unsuccessful recombination, missing word, reached the end of pairs
        case5_mpairs = [("i", "NN"), ("was", "VBD"), ("born", "VBN"), ("in", "IN"), ("hong", "NNP"), ("kong", "NN")]
        case5_gpairs = [("i", "NN"), ("was", "VBD"), ("born", "VBN"), ("in", "IN"), ("hong kong la", "NNP")]
        case5_pos = 4
        with self.assertRaises(MPOS.MissingWordException) as miss_wrd:
            MPOS.combine_adjacent(case5_mpairs, case5_gpairs, case5_pos, case5_pos)
        assert miss_wrd.exception.gold_wrd == case5_gpairs[case5_pos][0]
        assert miss_wrd.exception.model_wrd == "hong kong"


        # TODO: add case of multiple recombinations needed
        # TODO: add case of middle of recombination eg. "i(NN) was(VBP) in(IN) hong kong(NNP) for gon(NN) ba(NN) la(NN)"


if __name__ == '__main__':
    unittest.main()
