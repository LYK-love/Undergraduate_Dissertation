import csv
import json
import os

def _read_json_file(file_path):
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

def _save_jsonl_file(data, file_path):
    with open(file_path, 'w') as file:
        for obj in data:
            json_line = json.dumps(obj)
            file.write(json_line + '\n')

def write_label_to_dataset(csv_file_path, json_obj_list, labeled_dataset_path):
    '''
    根据给定的{csv_file}来给dataset_path中的样本加入标签。
    {csv_file}分为三列： idx, sample_id, label
    本方法会根据(idx,sample_id)匹配样本，并对其加上对应的label
    Args:
        csv_file_path:
        json_obj_list: 没有标注的代码-注释变更样本组成的数据集
        labeled_dataset_path: 标注后的数据集

    Returns:

    '''

    # 读取CSV文件
    csv_data = []
    with open(csv_file_path, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            csv_data.append(row)

    jsonl_data = json_obj_list

    # 创建一个字典，用于快速查找JSON对象
    json_dict = {}
    for json_obj in jsonl_data:
        idx = json_obj['idx']
        sample_id = json_obj['sample_id']
        json_dict[(idx, sample_id)] = json_obj

    # 遍历CSV文件的每一行，匹配JSON对象，并添加label属性
    for row in csv_data:
        idx = row['idx']
        sample_id = row['sample_id']
        label = row['label']

        # 查找对应的JSON对象
        json_obj = json_dict.get((idx, sample_id))

        # 如果在csv文件中找得到匹配的人工标注结果, 就把该结果添加到该样本
        if json_obj:
            # 更新JSON对象的属性
            json_obj['label'] = label

    # 将更新后的JSONL写回文件
    with open(labeled_dataset_path, 'w') as jsonl_file:
        for json_obj in jsonl_data:
            jsonl_file.write(json.dumps(json_obj) + '\n')

    check_if_all_have_label(labeled_dataset_path)



def check_if_all_have_label(labeled_dataset_path):
    '''
    检查JSONL文件中每个对象是否具有label属性，并且该属性的值是否都为True或False
    如果存在缺失, 则把缺失标记的样本标记为False
    Args:
        labeled_dataset_path:

    Returns:

    '''
    print(f"Checking if all samples have label. Dataset: {labeled_dataset_path}")
    all_have_label = True
    json_obj_list = _read_json_file(labeled_dataset_path)

    new_json_obj_list = [] # 新的数据集

    for json_obj in json_obj_list:
        if 'label' not in json_obj:
            all_have_label = False
            json_obj['label'] = False

            print(f"Error: Missing 'label' property in JSON object: {json_obj}")
        else:
            label = json_obj['label']
            if not isinstance(label, bool):
                all_have_label = False
                json_obj['label'] = False

                print(f"Error: Invalid 'label' value in JSON object: {json_obj}")

        new_json_obj_list.append(json_obj)


    if all_have_label == True: # 所有样本都没有问题, 不需要更改原来的数据集
        print(f"Label checked. All samples have label with bool value in dataset: {labeled_dataset_path}")
    else: # 需要把新的数据集写入原来的文件
        _save_jsonl_file(new_json_obj_list,labeled_dataset_path)

    return


def read_label_from_dataset_and_save(dataset_path, csv_file_path):
    '''
    从一个dataset中读取每个样本的label，并写入csv文件。
    {csv_file}分为三列： idx, sample_id, label
    本方法仅仅用于研究数据集。
    Args:
        dataset_path:
        saved_dataset_path:

    Returns:

    '''


    # 读取JSONL文件
    jsonl_data = _read_json_file(dataset_path)


    csv_rows = []
    for json_obj in jsonl_data:
        idx = json_obj['idx']
        sample_id = json_obj['sample_id']
        label = json_obj.get('label')

        row = [idx,sample_id,label]

        csv_rows.append(row)


    # 写入更新后的CSV文件
    with open(csv_file_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        header = ['idx', 'sample_id', 'label']
        csv_writer.writerow(header)  # 添加标题行
        csv_writer.writerows(csv_rows)


def manually_read_label_in_all_dataset_and_save(dataset_dir_path, save_dataset_dir_path):
    os.makedirs(save_dataset_dir_path, exist_ok=True)

    # 遍历子目录下的文件
    # file_path: full_valid.jsonl  mix_vocab.json  mix_vocab_embeddings.pkl  test.jsonl  train.jsonl  valid.jsonl  vocab.json  ....

    for file_name in os.listdir(dataset_dir_path):
        file_path = os.path.join(dataset_dir_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.jsonl'):
            file_name = file_name.replace('.jsonl', '.csv') # 现在以csv结尾
            save_file_path = os.path.join(save_dataset_dir_path, file_name)
            read_label_from_dataset_and_save(file_path,save_file_path)

if __name__ == "__main__":
    # "Project"文件夹的路径
    project_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subjects_dir_path = f"{project_dir_path}/subjects"

    dataset_dir_path = f'{subjects_dir_path}/data/labeled_dataset'
    csv_result_dir_path = f'{subjects_dir_path}/data/label_res'

    # manually_read_label_in_all_dataset_and_save(dataset_dir_path, csv_result_dir_path)

    unlabeled_dataset_file_path = f'/Users/lyk/Projects/MyOfficialProjects/Undergraduate_Thesis/Project/subjects/data/processed_dataset/test2.jsonl'
    labeled_dataset_file_path = f'{dataset_dir_path}/test2.jsonl'
    csv_result_file_path = f'{csv_result_dir_path}/test.csv'

    json_obj_list = _read_json_file(unlabeled_dataset_file_path)

    write_label_to_dataset(csv_result_file_path, json_obj_list, labeled_dataset_file_path)