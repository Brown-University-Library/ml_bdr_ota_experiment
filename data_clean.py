##############################################
# Broad Themes:
# Agriculture
# Armed Forces
# Commerce
# Corruption
# Courts
# Diplomatic Relations
# Dissenters
# Economic Assistance
# Economic Policy
# Education
# Geopolitics
# Industrial Relations
# Investments
# Medical Care
# Press
# Public Administration
# Religion
# Social Conditions
##############################################

import pandas as pd
import json
import pprint

def load_data():
    # Get the docs from the raw json file
    raw_data_dict = json.load(open('source_data/OtA_raw.json'))
    # pprint.pprint(raw_data_dict[0])
    # load the docs into a dataframe
    df = pd.DataFrame(raw_data_dict)
    return df



if __name__ == '__main__':
    df = load_data()
    # # print a list of the unique values in the mods_subject_broad_theme_ssim column
    # broad_themes = df['mods_subject_broad_theme_ssim'].dropna().values
    # broad_themes = [item for sublist in broad_themes for item in sublist]
    # for i in sorted(set(broad_themes)):
    #     print(i)