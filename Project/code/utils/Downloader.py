import json

def read_json_file_for_given_line(file_path, line_count):
    '''

    :param file_path:
    :return: list(dict), each dict is a json object
    '''
    data = []

    with open(file_path, "r") as file:
        for i, line in enumerate(file):
            if i >= line_count:
                break
            json_obj = json.loads(line)
            data.append(json_obj)
    return data

def save_jsonl_file(data, file_path):
    with open(file_path, 'w') as file:
        for obj in data:
            json_line = json.dumps(obj)
            file.write(json_line + '\n')

def download_file(download_data_path, save_file_path, line_count = 1000):
    data = read_json_file_for_given_line(download_data_path, line_count)
    save_jsonl_file(data, save_file_path)
    print(f"file download end. Downloaded {line_count} lines")


import os


def download_dataset_dir_by_given_ratio(download_dataset_dir_path, save_dataset_dir_path, ratio=0.2):
    print(f"Begin downloading. Original dataset dir path: {download_dataset_dir_path}. Target dateset dir path: {save_dataset_dir_path}, ratio: {ratio}")

    # 创建保存数据集的目录
    os.makedirs(save_dataset_dir_path, exist_ok=True)

    # 遍历下载数据集目录下的子目录
    for subdir in os.listdir(download_dataset_dir_path):
        '''
        original subdir_path: {data/cup2_dataset, data/cup2_updater_dataset}
        save subdir_path: {data/cup2_dataset, data/cup2_updater_dataset}
        '''
        original_subdir_path = os.path.join(download_dataset_dir_path, subdir)
        if os.path.isdir(original_subdir_path):
            # 创建对应子目录的保存目录
            save_subdir_path = os.path.join(save_dataset_dir_path, subdir)

            # If the target directory already exists, raise an OSError if exist_ok is False. Otherwise no exception is
            # raised.  This is recursive.
            os.makedirs(save_subdir_path, exist_ok=True)

            # 遍历子目录下的文件
            # file_path: full_valid.jsonl  mix_vocab.json  mix_vocab_embeddings.pkl  test.jsonl  train.jsonl  valid.jsonl  vocab.json  ....

            for file_name in os.listdir(original_subdir_path):
                file_path = os.path.join(original_subdir_path, file_name)
                if os.path.isfile(file_path) and file_name.endswith('.jsonl'):
                    save_file_path = os.path.join(save_subdir_path, file_name)
                    with open(file_path, 'r') as input_file, open(save_file_path, 'w') as output_file:
                        # 计算应下载的行数
                        num_lines = sum(1 for line in input_file)
                        download_lines = int(num_lines * ratio)

                        # 重置文件读取位置
                        input_file.seek(0)

                        # 下载指定行数的数据到新文件
                        for i in range(download_lines):
                            line = input_file.readline()
                            output_file.write(line)

                        print(f"Original dataset path: {file_path}. Original sample num: {num_lines}")
                        print(f"Download dataset path: {save_file_path}. Download sample num: {download_lines}")
                        print(f"Download ratio: {ratio}")



if __name__ == "__main__":
    '''
    Run locally
    '''

    project_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    subjects_dir_path = f"{project_dir_path}/subjects"
    data_dir_path = f"{subjects_dir_path}/data_for_test"
    dataset_dir_path = f'{data_dir_path}/raw_dataset'

    save_dataset_dir_path = dataset_dir_path

    is_local = False
    if is_local:
        download_dataset_dir_path = '/Users/lyk/Documents/Research/CUP2/data'
    else:
        download_dataset_dir_path = '/root/autodl-tmp/data'

    download_dataset_dir_by_given_ratio(download_dataset_dir_path, save_dataset_dir_path, ratio=0.0001)

