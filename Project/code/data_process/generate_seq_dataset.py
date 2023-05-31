import argparse
import json
from pathlib import Path

from Project.code.utils.FIle_process import read_json_file, save_jsonl_file


def generate_clean_dataset(old_data_file_dir_path, cleaned_data_file_dir_path):
    '''
    Generate_clean_dataset using 'id_delete.json', 'positive_id.json', 'positive_sample_id.json' in current dir.
    :param old_data_file_dir_path: dir of input old_data
    :param cleaned_data_file_dir_path: dir of output cleaned data
    :return:
    '''
    print("generate_clean_dataset START!")

    # splits = ['train', 'full_valid', 'valid', 'test']
    splits = ['train', 'full_valid', 'valid', 'test']

    try:
        with open('id_delete.json') as old_data_file:
            deleted_id = set(json.load(old_data_file))
    except:
        print('no id_delete.json')
        deleted_id = []
    try:
        with open('positive_id.json') as old_data_file:
            positive_id = set(json.load(old_data_file))
    except:
        print('no positive_id.json')
        positive_id = []
    try:
        with open('positive_sample_id.json') as old_data_file:
            positive_sample_id = set(json.load(old_data_file))
    except:
        print('no positive_sample_id.json')
        positive_sample_id = []
    for s in splits:

        cleaned_data_file_path = cleaned_data_file_dir_path + "/" + f'{s}_cleaned.jsonl'
        cleaned_data = []

        old_data_file_path = old_data_file_dir_path + "/" + f'{s}.jsonl'
        old_data = read_json_file(old_data_file_path)

        for json_obj in old_data:
            # #利用之前的去重结果"id_delete.json", 删掉不要的{id}_{sample_id}
            id = f"{json_obj['idx']}_{json_obj['sample_id']}"  #
            if id in deleted_id:
                continue

            # 修改label, 利用之前的label_correction结果'positive_id.json'
            if id in positive_id:
                json_obj['label'] = True

            # 利用之前的label_correction结果'positive_sample_id.json'
            sample_id = int(json_obj['sample_id'])
            json_obj['confidence'] = int(sample_id in positive_sample_id)
            cleaned_data.append(json_obj)
            save_jsonl_file(cleaned_data, cleaned_data_file_path)
    print("generate_clean_dataset END!")











if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_dir_path', required=True, type=str, help='path to original dataset')
    parser.add_argument('--output_file_dir_path', required=True, type=str, help='path to the cleaned dataset')
    args = parser.parse_args()
    old_data_path = Path(args.input_file_dir_path)
    new_data_path = Path(args.output_file_dir_path)
    # old_data_path = '/root/autodl-tmp/data/cup2_dataset'
    # new_data_path = '/root/autodl-tmp/data/cleaned_OcdData'
    generate_clean_dataset(old_data_path, new_data_path)
