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

## TODO: Find out how to more safely open file: so maybe not just texgrid with open and read, use with
class ParsedTextgrid:
    """ Parsed Textgrid Class:
        This class is the parsed textgrid
        Attributes: x_min_max, number of tiers, array of tiers"""

    def __init__(self, textgrid: str):
        self.textgrid = open(textgrid, 'r')
        self.x_min_max = _parse_x(self.textgrid)
        self.tier_num = _parse_tier_num(self.textgrid)
        self.tiers = _parse_tiers(self.tier_num, self.x_min_max, self.textgrid)
        assert self.tier_num == len(self.tiers)

    def get_x_min_max(self):
        return self.x_min_max

    def get_textgrid(self):
        return self.textgrid

    def get_tier_num(self):
        return self.tier_num

    def get_tiers(self):
        return self.tiers


class Tier:
    """ Tier class:
        This class represent a tier in the textgrid.
        Attributes: name, x_min_max, interval_num, intervals"""

    def __init__(self, x_min_max: (float, float), textgrid):
        self.name = _parse_tier_name(textgrid)
        self.x_min_max = x_min_max
        self.interval_num = _parse_interval_num(textgrid)
        self.intervals = _parse_intervals(self.interval_num, textgrid)
        assert self.interval_num == len(self.intervals)

    def get_name(self):
        return self.name

    def get_x_min_max(self):
        return self.x_min_max

    def get_interval_num(self):
        return self.interval_num

    def get_intervals(self):
        return self.intervals


class Interval:
    def __init__(self, textgrid):
        self.x_min_max = _parse_x(textgrid)
        self.text = _parse_text(textgrid)

    def get_x_min_max(self):
        return self.x_min_max

    def get_text(self):
        return self.text

    def set_text(self, new_text: str):
        self.text = new_text


def _parse_tier_num(textgrid):
    """Function parses the text grid and returns the number of tiers"""
    tiers_str = 'tiers? '
    size_str = 'size = '
    size_strlen = len(size_str)
    size = None

    for line in textgrid:
        tiers_pos = line.find(tiers_str)

        # if the tiers? is in this line, the size should be next line
        if tiers_pos != -1:
            size_line = textgrid.readline()
            size_pos = size_line.find(size_str)
            line_length = len(size_line)
            size = int(size_line[size_pos + size_strlen:line_length])
            break

    return size


def _parse_tiers(tier_num, x_min_max, textgrid):
    """Function parses the text grid and returns a list of Tier objects"""
    tiers = []
    for i in range(tier_num):
        tiers.append(Tier(x_min_max, textgrid))

    return tiers


def _parse_interval_num(textgrid):
    """Function parses the textgrid and returns the number of intervals in a tier"""
    interval_num_str = "intervals: size = "
    interval_num = None

    for line in textgrid:
        interval_num_pos = line.find(interval_num_str)
        line_length = len(line)

        if interval_num_pos != -1:
            interval_num = int(line[interval_num_pos + len(interval_num_str):line_length].strip())
            break
    # print(interval_num)
    return interval_num


def _parse_tier_name(textgrid):
    """Function parses the textgrid and returns the tier name"""
    name_str = "name = \""
    name = None

    for line in textgrid:
        name_pos = line.find(name_str)
        line_length = len(line)

        if name_pos != -1:
            name = line[name_pos + len(name_str):line_length].replace('\n', '').replace('\"', '').strip()
            break
    # print(name)
    return name


def _parse_intervals(interval_num, textgrid):
    """Function parses the textgrid and returns the intervals in each tier"""
    intervals = []
    for i in range(interval_num):
        intervals.append(Interval(textgrid))

    return intervals


def _parse_text(textgrid):
    """Function parses the textgrid and returns the text of an interval"""
    text_string = "text = \""
    text = None

    for line in textgrid:
        text_pos = line.find(text_string)
        line_length = len(line)

        if text_pos != -1:
            text = line[text_pos + len(text_string):line_length].replace('\n', '').replace('\"', '').strip()
            break
    # print(text)
    return text


def _parse_x(textgrid):
    """Function parses the text grid and returns the x min and max for interval"""
    x_min_str = "xmin = "
    x_min_strlen = len(x_min_str)
    x_max_str = "xmax = "
    x_max_strlen = len(x_max_str)
    x_min = None
    x_max = None

    for line in textgrid:
        # line = line.decode("ascii")
        xmin_pos = line.find(x_min_str)
        xmax_pos = line.find(x_max_str)
        line_length = len(line)

        if xmin_pos != -1:
            x_min = float(line[xmin_pos + x_min_strlen:line_length].strip())

        if xmax_pos != -1:
            x_max = float(line[xmax_pos + x_max_strlen:line_length].strip())

        if x_min is not None and x_max is not None:
            break

    return x_min, x_max
