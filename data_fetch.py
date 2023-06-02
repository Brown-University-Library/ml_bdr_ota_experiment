from urllib import request
import json

api_base_url = 'https://repository.library.brown.edu/api/search/?q='
query_parameters = '%20AND%20'.join([
                    r'rel_is_member_of_collection_ssim:"bdr:318399"',
                    '-rel_is_part_of_ssim:*',
                    'ds_ids_ssim:MODS'
                    ])



def get_data_from_api():
    # Get the docs from the api, paging through the results
    results = []
    start = 0
    while True:
        url = api_base_url + query_parameters + '&rows=500&start=' + str(start)
        print(url)
        response = request.urlopen(url)
        raw_json = json.loads(response.read())
        docs = raw_json['response']['docs']
        results.extend(docs)
        if len(docs) < 500:
            break
        start += 500
    print('\n\nTotal docs: ', len(results))
    return results

def save_data_to_file():
    results = get_data_from_api()
    with open('source_data/OtA_raw.json', 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    save_data_to_file()