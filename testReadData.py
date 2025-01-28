import unittest
import Read_Data as rd

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


if __name__ == '__main__':
    unittest.main()
