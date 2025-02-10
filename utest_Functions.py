import unittest
import Read_Data as rd
import numpy as np
import pandas as pd
import Multilingual_POS as m_pos

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


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.filenames = []
        for file in FILES:
            self.filenames.append(PATH + file)

    def test_tier_interval_nums(self):
        # test
        for filename in self.filenames:
            parsed_ts = rd.ParsedTextgrid(filename)
            tier_nums = parsed_ts.get_tier_num()
            tiers = parsed_ts.get_tiers()
            self.assertEqual(tier_nums, len(tiers))

            for tier in tiers:
                interval_nums = tier.get_interval_num()
                intervals = tier.get_intervals()
                self.assertEqual(interval_nums, len(intervals))


    def test_POS_tagger(self):
        # test POS tagger
        # # test recombine
        # tks = ["I", "am", "a", "&", "and", "the"]
        # recombined = recombine(tks, "&")
        # print(recombined)
        #
        # tks2 = ["I", "&", "a", "&", "and", "the"]
        # recombined2 = recombine(tks2, "&")
        # print(recombined2)
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


if __name__ == '__main__':
    unittest.main()
