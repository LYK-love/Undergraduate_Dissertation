import json
import random


from Project.code.CommentSimilarity.SentenceCloneType import SentenceCloneType
from Project.code.data_process.deduplicate import deduplicate
from Project.code.data_process.generate_cleaned_dataset import generate_cleaned_dataset_and_save
from Project.code.data_process.label_correction import label_correction
from Project.code.utils.Manually_label import write_label_to_dataset
from Project.code.utils.ZeroShot import add_classification_property_using_model
from Project.code.utils.sentence_clone_detection import detect_sentence_clone


# def read_json_file(filename, num_lines):
#     with open(filename, 'r') as file:
#         lines = []
#         for i, line in enumerate(file):
#             if i >= num_lines:
#                 break
#             json_obj = json.loads(line)
#             lines.append(json_obj)
#         return lines

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

def get_line_cnt(file_path):
    '''

    :param file_path:
    :return: list(dict), each dict is a json object
    '''
    line_cnt = 0
    with open(file_path, 'r') as file:
        for line in file:
            line_cnt+=1
    return line_cnt

def get_dataset_ratio(training_dataset_path, test_dataset_path):
    '''
    Get the ratio of training_dataset  and test_dataset, should be 4 : 1 ( return 0.8)
    Args:
        training_dataset_path:
        test_dataset_path:

    Returns:

    '''
    size_of_training_dataset = get_line_cnt(training_dataset_path)
    size_of_test_dataset = get_line_cnt(test_dataset_path)

    size_of_total = size_of_training_dataset + size_of_test_dataset


    ratio = size_of_training_dataset / size_of_total
    return ratio

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

def rewrite_jsonl_file(input_file_path, output_file_path, num_lines):
    '''
    将原始的 JSONL 文件的前 M 行重新写入到同一文件中：
    Args:
        input_file_path:
        output_file_path:
        num_lines:

    Returns:

    '''
    new_data = read_json_file_for_given_line(input_file_path, num_lines)
    save_jsonl_file(new_data,output_file_path)



# 给每个JSON对象添加 classification 属性, 这个方法仅仅用做演示. 实际使用的是CPC的代码.
def add_classification_property(json_obj_list):
    '''
    :param json_obj_list: list(dict), each dict is a json object
    :return:
    '''
    # ---------------------

    use_model = False
    if use_model:
        # Use ZeroShot to implement CPC's comment classification. Because it's extremely slow, so during the test/dubug procudure,
        # we don;t use it and use a mock method alternatively.
        # json_obj_list = json_obj_list[:10]
        json_obj_list = add_classification_property_using_model(json_obj_list)
        # ---------------------
    else:
        for json_obj in json_obj_list:
            classification = {
                'what': random.choices([0, 1], weights=[0.8, 0.2])[0],
                'why': random.choices([0, 1], weights=[0.95, 0.05])[0],
                'how-it-is-done': random.choices([0, 1], weights=[0.6, 0.4])[0],
                'how-to-use': random.choices([0, 1], weights=[0.85, 0.15])[0],
                'property': random.choices([0, 1], weights=[0.2, 0.8])[0]
            }

            # 不允许全1或全0的情况
            if all(value == 0 for value in classification.values()):
                classification['what'] = 1
            elif all(value == 1 for value in classification.values()):
                classification['what'] = 0
                classification['why'] = 0
                classification['how-it-is-done'] = 0
                classification['how-to-use'] = 0
                classification['property'] = 1

            json_obj["classification"] = classification

    return json_obj_list

def add_sentence_level_clone_type(json_obj_list):
    '''
    add "clone_type" attribute to each json_obj.
    "clone_type":
    {
        "textual": 0
        "lexical": 1
        "syntactic": 0
        "semantic": 0
    }

    :param json_obj_list:
    :return:
    '''
    for json_obj in json_obj_list:


        is_type_1_clone = 0
        is_type_2_clone = 0
        is_type_3_clone = 0
        is_type_4_clone = 0

        is_clone = False
        is_clone = detect_sentence_clone(json_obj, SentenceCloneType(1))
        if is_clone:
            is_type_1_clone = 1
        else:
            is_clone = detect_sentence_clone(json_obj, SentenceCloneType(2))
            if is_clone:
                is_type_2_clone = 1
            else:
                is_clone = detect_sentence_clone(json_obj, SentenceCloneType(3))
                if is_clone:
                    is_type_3_clone = 1
                else:
                    is_type_4_clone = random.randint(0, 1)

        sentence_clone_type = {
            "textual": is_type_1_clone,
            "lexical": is_type_2_clone,
            "syntactic": is_type_3_clone,
            "semantic": is_type_4_clone
        }

        json_obj["clone_type"] = sentence_clone_type

    return json_obj_list

def look_classification(processed_data_file_path):
    processed_data = read_json_file(processed_data_file_path)
    for single_sample in processed_data:
        classification_info = single_sample['classification'] # {what: 0, how: 1, why: 0}
        print(single_sample)

        for key,value  in classification_info.items():
            print(f'{key}: {value}; ', end="")
        print()

def save_jsonl_file(data, file_path):
    with open(file_path, 'w') as file:
        for obj in data:
            json_line = json.dumps(obj)
            file.write(json_line + '\n')


def modify_and_clean_dataset(original_dataset_dir, original_update_dataset_dir, cleaned_dataset_dir, id_deleted_dir_path, sample_id_files_dir_path ):
    '''
    original_update_dataset_dir 只是被用作label_correction，不会作为数据集的一部分。
    Args:
        original_dataset_dir:
        original_update_dataset_dir:
        cleaned_dataset_dir:
        id_deleted_dir_path:
        sample_id_files_dir_path:

    Returns:

    '''
    deduplicate(original_dataset_dir,id_deleted_dir_path)#向{id_deleted_dir_path}目录下输出文件
    label_correction(original_update_dataset_dir, sample_id_files_dir_path)# 向{sample_id_files_dir_path}目录下输出文件
    generate_cleaned_dataset_and_save(original_dataset_dir, cleaned_dataset_dir, id_deleted_dir_path, sample_id_files_dir_path)#利用{id_deleted_dir_path}, {sample_id_files_dir_path}目录下的文件作为规则来过滤数据集
    print(f"Dataset Cleaning End. Cleaned dataset is saved in dir {cleaned_dataset_dir}")



def process_sentence_and_save(file_path, classified_file_path, clone_detected_file_path):
    data = read_json_file(file_path)

    #注释的语句分类
    classified_data = add_classification_property(data)
    save_jsonl_file(classified_data, classified_file_path)
    print(f'Classification end. Save data to {classified_file_path}')

    # 注释的语句的克隆检测
    clone_detected_data = add_sentence_level_clone_type(classified_data)
    save_jsonl_file(clone_detected_data, clone_detected_file_path)
    print(f'Clone detection end. Save data to {clone_detected_file_path}')
    print("Sentence Processing End.")

def process_sentence(file_path):
    data = read_json_file(file_path)

    #注释的语句分类
    classified_data = add_classification_property(data)
    print(f'Classification end. Data not saved')

    # 注释的语句的克隆检测
    clone_detected_data = add_sentence_level_clone_type(classified_data)
    print(f'Clone detection end. Data not saved')

    print("Sentence Processing End.")
    return clone_detected_data



def change_label_use_heuristic_using_path(dataset_path):
    samples = read_json_file(dataset_path)
    changed_samples = change_label_use_heuristic(samples)
    return changed_samples

def change_label_use_heuristic(json_obj_list):
    '''
    如果原来的label是True(Positive, 过时注释. 说明两个注释不相似, 不存在注释克隆), 但是检查出来property==True(即该注释的property类型存在克隆)👎clone_type = “lexical”, 那么就翻转注释的label为False, 认为存在注释克隆.
    对"how-it-is-done"和"why"同理, 只是概率小一些

    "clone_type":
    {
        "textual": 0
        "lexical": 1
        "syntactic": 0
        "semantic": 0
    }

    Args:
        json_obj_list:

    Returns:

    '''
    json_obj_list = make_samples_balance(json_obj_list)



    # reversed_labels_cnt = 0 # 被启发式规则更改label为False的注释的数量
    # for json_obj in json_obj_list:
    #     clone_type = json_obj['clone_type']
    #     classification = json_obj['classification']
    #     label = json_obj['label']
    #
    #
    #     if classification['property'] == True and clone_type['lexical'] == 1:
    #         label = False
    #         reversed_labels_cnt += 1
    #     elif classification["how-it-is-done"] == True or classification["why"] == True:
    #         random_num = random.random()
    #
    #         # 指定执行函数的概率
    #         probability = 0.01  # 1%的概率执行函数
    #
    #         # 如果随机数小于等于概率，则执行函数
    #         if random_num <= probability:
    #             label = False
    #             reversed_labels_cnt += 1
    #     json_obj['label'] = label
    #
    # input_sample_num = len(json_obj_list)
    # mutation_rate = reversed_labels_cnt/input_sample_num
    # print(f'Original sample num: {input_sample_num}')
    # print(f"被启发式规则更改label为False的注释的数量: {reversed_labels_cnt}")
    # print(f"Mutation rate: {mutation_rate}")
    return json_obj_list


def make_samples_balance(json_obj_list):
    '''
    人工对数据集打标签， 该方法只是作为一个mock, 实际的过程是手工完成的
    Args:
        json_obj_list:

    Returns:

    '''
    # 指定执行函数的概率
    probability = 0.005  # 0.01%的概率执行函数
    reversed_labels_cnt = 0 # 被随机规则更改label为False的注释的数量

    for json_obj in json_obj_list:
        if json_obj['label'] == False:
            random_num = random.random()
            # 如果随机数小于等于概率，则执行函数
            if random_num <= probability:
                json_obj['label'] = True
                reversed_labels_cnt += 1

    input_sample_num = len(json_obj_list)
    reverse_rate = reversed_labels_cnt / input_sample_num
    print(f'Original sample num: {input_sample_num}')
    print(f"被启发式规则更改label为True的注释的数量: {reversed_labels_cnt}")
    print(f"Reverse rate: {reverse_rate}")
    return json_obj_list






def add_label(sample_data_file_path,  labeled_file_path, label_csv_file_path = ''):


    samples = read_json_file(sample_data_file_path)

    # 读取存储了人工标注结果的csv文件, 对样本进行人工标注。该方法在debug阶段不被采用。
    write_label_to_dataset(label_csv_file_path, samples, labeled_file_path)


    # changed_samples = change_label_use_heuristic(samples)
    # save_jsonl_file(changed_samples, labeled_file_path)

    print(f'Manually Label Result is saved to the dataset. Save data to {labeled_file_path}')




# def change_label_use_heuristic_and_save_using_data(samples, changed_file_path):
#     '''
#     以数据而非数据的路径作为输入
#     Args:
#         samples:
#         changed_file_path:
#
#     Returns:
#
#     '''
#     changed_samples = change_label_use_heuristic(samples)
#     save_jsonl_file(changed_samples, changed_file_path)
#
#     print(f'Comment Label Reversion End. Save data to {changed_file_path}')


def aggregate_comments(sample_data_file_path):
    samples = read_json_file(sample_data_file_path)
    comments = merge_json_objects(samples)
    print(f'Comment Aggregation End. Data not saved')

    print(f'original sample num: {len(samples)}')
    print(f'aggregated comment num: {len(comments)}')
    return comments



def merge_samples_and_save(sample_data_file_path, merged_file_path):
    samples = read_json_file(sample_data_file_path)
    merge_samples_and_save_using_data(samples,merged_file_path)
    return

def merge_samples_and_save_using_data(samples, merged_file_path):
    merged_samples = merge_json_objects(samples)

    save_jsonl_file(merged_samples, merged_file_path)

    print(f'Dataset Merge End. Save data to {merged_file_path}')
    print(f'Original sample num: {len(samples)}')
    print(f'Merged sample num: {len(merged_samples)}')

    return

def merge_json_objects(json_obj_list):
    merged_objects = []
    object_dict = {}

    for obj in json_obj_list:
        idx = obj["idx"]
        sample_id = obj["sample_id"]

        # 构建合并后的对象
        if idx in object_dict and sample_id in object_dict[idx]:
            # 合并 src_desc 和 src_desc_tokens 属性
            merged_obj = object_dict[idx][sample_id]
            merged_obj["src_desc"] += obj["src_desc"]
            merged_obj["src_desc_tokens"] += obj["src_desc_tokens"]

            # 加入"composite_true_label_num"属性， 统计该合成的评论究竟由几个Negative样本(即具备第二类 CUP Comment clone的样本)组成。
            # Comment的"label"值取组成的样本中数量较多的那个label. 即： 如果组成的样本中大多数都是negative(False), 则该合成的comment就是negative(False)的
            if not merged_obj.has_key("composite_false_label_num"):
                merged_obj["composite_false_label_num"] = 0
            if not merged_obj.has_key("total_label_num"):
                merged_obj["total_label_num"] = 0

            merged_obj["total_label_num"] += 1
            if obj["label"] == False:
                merged_obj["composite_false_label_num"] += 1
            false_ratio = merged_obj["composite_false_label_num"] / merged_obj["total_label_num"]
            if false_ratio >= 0.5:
                merged_obj["label"] = False
            else:
                merged_obj["label"] = True
        else:
            merged_obj = obj

        # 更新字典中的对象
        if idx not in object_dict:
            object_dict[idx] = {}
        object_dict[idx][sample_id] = merged_obj

    # 将合并后的对象添加到列表中
    for idx_dict in object_dict.values():
        merged_objects.extend(list(idx_dict.values()))

    return merged_objects

def download_file(download_data_path, save_file_path, line_count = 100):
    data = read_json_file_for_given_line(download_data_path, line_count)
    save_jsonl_file(data, save_file_path)
    print(f"file download end. Downloaded {line_count} lines")

if __name__ == "__main__":
    pass
    # file_name = 'test'
    # input_file_dir = '../data'
    #
    # output_file_dir = '../data'
    #
    #
    # train_data_path = f'{input_file_dir}/{file_name}.jsonl'
    # classified_file_path = f'{output_file_dir}/{file_name}_classified.jsonl'
    # clone_detected_file_path = f'{output_file_dir}/{file_name}_detected.jsonl'
    # merged_file_path = f'{output_file_dir}/{file_name}_merged.jsonl'
    #
    # final_processed_file_path = clone_detected_file_path
    #
    #
    #
    # process_sentence(train_data_path, classified_file_path, clone_detected_file_path)
    # aggregate_comments_and_save(clone_detected_file_path, merged_file_path )
    #
    #
    # look_classification(merged_file_path)
