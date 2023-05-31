import argparse
import json
import os
from pathlib import Path

path = os.path.dirname(__file__) # 先找到当前文件 所在的目录
# path = os.path.dirname(path)     # 往上倒一层目录,也就是 config.txt 所在的文件夹


def generate_cleaned_dataset_and_save(old_data_file_dir_path, cleaned_data_file_dir_path, id_deleted_dir_path, sample_id_files_dir_path):
    '''
    Generate_clean_dataset using 'id_delete.json', 'positive_id.json', 'positive_sample_id.json' in current dir.
    :param old_data_file_dir_path: dir of input old_data
    :param cleaned_data_file_dir_path: dir of output cleaned data
    :return:
    '''
    print("generate_clean_dataset START!")
    # print(os.getcwd())
    #
    # id_delete_file_path = os.path.join(path, 'id_delete.json')  # 拼接文件的路径
    # positive_id_file_path = os.path.join(path, 'positive_id.json')
    # positive_sample_id_file_path = os.path.join(path, 'positive_sample_id.json')
    id_deleted_dir_path = Path(id_deleted_dir_path)
    sample_id_files_dir_path = Path(sample_id_files_dir_path)

    id_delete_file_path = id_deleted_dir_path / 'id_delete.json'  # 拼接文件的路径
    positive_id_file_path = sample_id_files_dir_path / 'positive_id.json'
    positive_sample_id_file_path = sample_id_files_dir_path / 'positive_sample_id.json'

    # splits = ['train', 'full_valid', 'valid', 'test']
    splits = ['train', 'full_valid',  'test'] #不存在'valid',因为输入的数据是来自cup2_dataset, 里面只有 ['train', 'full_valid', 'test']

    try:
        with open(id_delete_file_path) as old_data_file:
            deleted_id = set(json.load(old_data_file))
    except:
        print('no id_delete.json')
        deleted_id = []
    try:
        with open(positive_id_file_path) as old_data_file:
            positive_id = set(json.load(old_data_file))
    except:
        print('no positive_id.json')
        positive_id = []
    try:
        with open(positive_sample_id_file_path) as old_data_file:
            positive_sample_id = set(json.load(old_data_file))
    except:
        print('no positive_sample_id.json')
        positive_sample_id = []
    for file_name in splits:

        old_data_file_path = old_data_file_dir_path + "/" + f'{file_name}.jsonl'
        old_data = read_json_file(old_data_file_path)

        cleaned_data_file_path = cleaned_data_file_dir_path + "/" + f'{file_name}.jsonl'
        cleaned_data = []



        for json_obj in old_data:
            # #利用之前的去重结果"id_delete.json", 删掉不要的{id}_{sample_id}
            id = f"{json_obj['idx']}_{json_obj['sample_id']}"  #
            if id in deleted_id:
                continue

            #修改label, 利用之前的label_correction结果'positive_id.json'
            if id in positive_id:
                json_obj['label'] = True

            # 利用之前的label_correction结果'positive_sample_id.json'
            sample_id = int(json_obj['sample_id'])
            json_obj['confidence'] = int(sample_id in positive_sample_id)
            cleaned_data.append(json_obj)


        save_jsonl_file(cleaned_data, cleaned_data_file_path)
        print(f"Raw dataset name: {file_name}, path: {old_data_file_path}. Num of samples:{len(old_data)} ")
        print(f"Cleaned dataset name: {file_name}, path: {cleaned_data_file_path}. Num of samples:{len(cleaned_data)} ")
        print(f"Save cleaned dataset to: {cleaned_data_file_path}")

    print("generate_clean_dataset END!")


def read_json_file(file_path):
    '''

    :param file_path:
    :return: list(dict), each dict is a json object
    '''
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data

def save_jsonl_file(data, file_path):
    with open(file_path, 'w') as file:
        for obj in data:
            json_line = json.dumps(obj)
            file.write(json_line + '\n')






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_dir_path', required=True, type=str, help='path to original dataset')
    parser.add_argument('--output_file_dir_path', required=True, type=str, help='path to the cleaned dataset')
    args = parser.parse_args()
    old_data_path = Path(args.input_file_dir_path)
    new_data_path = Path(args.output_file_dir_path)
    # old_data_path = '/root/autodl-tmp/data/cup2_dataset'
    # new_data_path = '/root/autodl-tmp/data/cleaned_OcdData'
    generate_cleaned_dataset_and_save(old_data_path, new_data_path)
