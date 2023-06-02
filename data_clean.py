import pandas as pd
import json

def load_data():
    # Get the docs from the raw json file
    raw_json = json.load(open('source_data/OtA_raw.json'))
    docs = raw_json['response']['docs'] # list of dictionaries
    # load the docs into a dataframe
    df = pd.DataFrame(docs)
    print(df.describe())

if __name__ == '__main__':
    load_data()