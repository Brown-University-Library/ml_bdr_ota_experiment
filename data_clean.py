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

import tensorflow as tf

def load_data():
    # Get the docs from the raw json file
    raw_data_dict = json.load(open('source_data/OtA_raw.json'))
    # pprint.pprint(raw_data_dict[0])
    # load the docs into a dataframe
    df = pd.DataFrame(raw_data_dict)
    return df

def df_to_dataset(df, shuffle=True, batch_size=32):
    # Convert the dataframe into a tf.data.Dataset
    df = df.copy()
    labels = df.pop('mods_subject_broad_theme_ssim')
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


if __name__ == '__main__':
    df = load_data()
    # # print a list of the unique values in the mods_subject_broad_theme_ssim column
    # broad_themes = df['mods_subject_broad_theme_ssim'].dropna().values
    # broad_themes = [item for sublist in broad_themes for item in sublist]
    # for i in sorted(set(broad_themes)):
    #     print(i)

# create a copy of the dataframe with only the columns we want to use
    df = df[['pid','genre','keyword', 'mods_location_physical_location_ssim', 'mods_language_code_ssim', 'mods_subject_broad_theme_ssim']]
    # drop rows with no values in the mods_subject_broad_theme_ssim column
    df = df.dropna(subset=['mods_subject_broad_theme_ssim'])
    