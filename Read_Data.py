import numpy as np
import pandas as pd
import nltk

# read the csv files and figure out what to do with them
# raw_data = pd.read_csv('SpICE data/spice_voicesauce_raw.csv')
# print(raw_data.head())
# print(raw_data.describe())
# print(raw_data.shape)

# need to read the annotation data from the text_grid
# from raw data file, looks like an array with item[0] correpsonding to the first answer
# they have annotations there, do they have the questions?


# write parser to parse textgrid into a pandas of each tier and
class ParsedTextgrid:
    """ Parsed Textgrid Class:
        This class is the parsed textgrid
        Attributes: x_min_max, number of tiers, array of tiers"""
    def __init__(self, textgrid: str):
        self.textgrid = open(textgrid, 'r')
        self.x_min_max = self._parse_x()
        self.tier_num = self._parse_tier_num()
        self.tiers = self._parse_tiers()
        assert self.tier_num == len(self.tiers)

    def get_x_min_max(self):
        return self.x_min_max

    def get_textgrid(self):
        return self.textgrid

    def get_tier_num(self):
        return self.tier_num

    def get_tiers(self):
        return self.tiers

    def _parse_x(self):
        """Function parses the text grid and returns the x min and max"""
        x_min_str = "xmin = "
        x_min_strlen = len(x_min_str)
        x_max_str = "xmax = "
        x_max_strlen = len(x_max_str)
        x_min = None
        x_max = None

        for line in self.textgrid:
            xmin_pos = line.find(x_min_str)
            xmax_pos = line.find(x_max_str)
            line_length = len(line) - 2

            if xmin_pos != -1:
                x_min = float(line[xmin_pos + x_min_strlen:line_length])

            if xmax_pos != -1:
                x_max = float(line[xmax_pos + x_max_strlen:line_length])

            if x_min is not None and x_max is not None:
                break

        return x_min, x_max

    def _parse_tier_num(self):
        """Function parses the text grid and returns the number of tiers"""
        tiers_str = 'tiers? '
        size_str = 'size = '
        size_strlen = len(size_str)
        size = None

        for line in self.textgrid:
            tiers_pos = line.find(tiers_str)

            # if the tiers? is in this line, the size should be next line
            if tiers_pos != -1:
                size_line = self.textgrid.readline()
                size_pos = size_line.find(size_str)
                line_length = len(size_line)
                size = int(size_line[size_pos + size_strlen:line_length])
                break

        return size

    def _parse_tiers(self):
        """Function parses the text grid and returns a list of Tier objects"""
        tiers = []
        for i in range(self.tier_num):
            tiers.append(Tier(self.x_min_max, self.textgrid))

        return tiers


class Tier:
    """ Tier class:
        This class represent a tier in the textgrid.
        Attributes: name, x_min_max, interval_num, intervals"""
    def __init__(self, x_min_max: (float, float), textgrid):
        self.name = self._parse_tier_name(textgrid)
        self.x_min_max = x_min_max
        self.interval_num = self._parse_interval_num(textgrid)
        self.intervals = self._parse_intervals(textgrid)
        assert self.interval_num == len(self.intervals)

    def get_name(self):
        return self.name

    def get_x_min_max(self):
        return self.x_min_max

    def get_interval_num(self):
        return self.interval_num

    def get_intervals(self):
        return self.intervals

    def _parse_tier_name(self, textgrid):
        """Function parses the textgrid and returns the tier name"""
        name_str = "name = \""
        name = None

        for line in textgrid:
            name_pos = line.find(name_str)
            line_length = len(line) - 3

            if name_pos != -1:
                name = line[name_pos + len(name_str):line_length]
                break
        # print(name)
        return name

    def _parse_interval_num(self, textgrid):
        """Function parses the textgrid and returns the number of intervals in a tier"""
        interval_num_str = "intervals: size = "
        interval_num = None

        for line in textgrid:
            interval_num_pos = line.find(interval_num_str)
            line_length = len(line) - 2

            if interval_num_pos != -1:
                interval_num = int(line[interval_num_pos + len(interval_num_str):line_length])
                break
        # print(interval_num)
        return interval_num

    def _parse_intervals(self, textgrid):
        """Function parses the textgrid and returns the intervals in each tier"""
        intervals = []
        for i in range(self.interval_num):
            intervals.append(Interval(self.x_min_max, textgrid))

        return intervals


class Interval:
    def __init__(self, x_min_max: (float, float), textgrid):
        self.x_min_max = x_min_max
        self.text = self._parse_text(textgrid)

    def get_x_min_max(self):
        return self.x_min_max

    def get_text(self):
        return self.text

    def _parse_text(self, textgrid):
        """Function parses the textgrid and returns the text of an interval"""
        text_string = "text = \""
        text = None

        for line in textgrid:
            text_pos = line.find(text_string)
            line_length = len(line) - 3

            if text_pos != -1:
                text = line[text_pos + len(text_string):line_length]
                break
        # print(text)
        return text
