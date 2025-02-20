import numpy as np
import pandas as pd
import nltk
import Read_Data as Read
import Multilingual_POS as MPos

# now need to add the pos tags as a Tier into the texrgrid
# first take the textgrid, read it, change into the classes
# then use convenience tier to do the pos tag, do not remove the empty lines
HEADER = ["File type = \"ooTextFile\"\n",
          "Object class = \"TextGrid\"\n",
          "\n"]
INDENT = "    "


def write_data(parsed: Read.ParsedTextgrid, filename: str) -> None:
    """ This function takes a parsed text_grid with its pos tagged sentences
    and writes it to a text_grid file"""
    f = open(filename, "w")
    # write header
    f.writelines(HEADER)

    # from the parsed_tg
    # write x_min and x_max
    x_min_max = parsed.get_x_min_max()
    x_min = x_min_max[0]
    x_max = x_min_max[1]
    f.write("xmin = " + str(x_min) + " \n")
    f.write("xmax = " + str(x_max) + " \n")

    # get tier_nums
    tier_nums = parsed.get_tier_num()
    if tier_nums > 0:
        f.write("tiers? <exists> \n")
        f.write("size = " + str(tier_nums) + " \n")
        f.write("item []: \n")
        write_tiers(f, parsed.get_tiers())
        
    f.close()


def write_tiers(f, tiers):
    """This function takes the tiers list and writes it into textgrid"""
    tier_num = 1

    # remember to add a tier of POS!
    for tier in tiers:
        f.write(INDENT + f'item [{tier_num}]:\n')
        f.write(INDENT + INDENT + "class = \"IntervalTier\" \n")
        f.write(INDENT + INDENT + f'name = \"{tier.get_name()}\" \n')
        f.write(INDENT + INDENT + f'xmin = {str(tier.get_x_min_max()[0])} \n')
        f.write(INDENT + INDENT + f'xmax = {str(tier.get_x_min_max()[1])} \n')
        f.write(INDENT + INDENT + f'intervals: size = {str(tier.get_interval_num())} \n')

        intervals = tier.get_intervals()
        if tier.get_name() == "convenience-IU":
            tag_intervals(intervals)

        write_intervals(f, intervals)
        tier_num += 1


def write_intervals(f, intervals):
    """This function takes the intervals list and writes them into textgrid"""
    interval_num = 1
    
    for interval in intervals:
        f.write(INDENT + INDENT + f'intervals [{interval_num}]:\n')
        f.write(INDENT + INDENT + INDENT + f'xmin = {str(interval.get_x_min_max()[0])} \n')
        f.write(INDENT + INDENT + INDENT + f'xmax = {str(interval.get_x_min_max()[1])} \n')
        f.write(INDENT + INDENT + INDENT + f'text = \"{interval.get_text()}\" \n')

        interval_num += 1


def tag_intervals(intervals):
    """This function takes the intervals list of convience-IU and changes the text to be POS tagged text"""
    sym = ["&", "@", "＠"]
    disf = ["&", "mmm", "mnmm", "mmhm", "mm", "ahh", "huh", "umm", "mhm", "um"]
    for interval in intervals:
        text = interval.get_text()
        tagged, _ = MPos.POS_sentence(text)
        cleaned = MPos.recombine_and_retag(tagged, sym, disf)

        tagged_text = ""
        for pair in cleaned:
            tagged_text = tagged_text + pair[0] + f'({pair[1]}) '

        interval.set_text(tagged_text)
