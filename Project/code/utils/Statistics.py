from Project.code.utils.FIle_process import read_json_file


def analysis(dataset_path, analysis_result_file_path):
    '''
    给定一个数据集, 分析其内容
    Args:
        dataset_path: path of one json data file

    Returns:

    '''
    print(f"Analyzing dataset: {dataset_path}")

    account_classification_results(dataset_path, analysis_result_file_path)
    account_clone_detection_results(dataset_path, analysis_result_file_path)
    account_sample_num(dataset_path, analysis_result_file_path)
def account_sample_num(file_path, output_file_path):
    '''
    True( Positive): 代码有改动, 注释也有改动. 即不存在注释克隆
    False( Negative): 代码有改动, 注释没有改动. 即存在注释克隆

    :param file_path:
    :return:
    '''
    json_obj_list = read_json_file(file_path)
    sample_counts = 0

    object_counts = {"True": 0, "False": 0} # True

    for json_obj in json_obj_list:
        sample_counts += 1
        if json_obj["label"] == True or json_obj["label"] == 1:
            object_counts["True"] += 1
        else:
            object_counts["False"] += 1

    print("\nComment Label Counts:")

    pos_count = object_counts["True"]
    neg_count = object_counts["False"]
    total_count = pos_count+neg_count

    if output_file_path:
        with open(output_file_path, 'a') as f:
            f.write("\nComment Label Counts:\n")
            for attribute, count in object_counts.items():
                f.write(f"Num of {attribute} samples: {count}\n")

            if neg_count == 0:
                f.write("All samples are negative( False )")
            else:
                f.write(f"The negative rate: {neg_count / total_count}")

    else:
        for attribute, count in object_counts.items():
            print(f"Num of {attribute} samples: {count}")

        if neg_count == 0:
            print("All samples are negative( False )")
        else:
            print(f"The negative rate: {neg_count / total_count}")

    return

def account_classification_results(classified_file_path,output_file_path):
    json_obj_list = read_json_file(classified_file_path)


    # 统计属性出现次数和对象的属性数量分布
    attribute_counts = {'what': 0, 'why': 0, 'how-it-is-done': 0, 'how-to-use': 0, 'property': 0}
    object_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for json_obj in json_obj_list:
        classification = json_obj['classification']
        attributes = classification.keys()
        comment_type_count_of_sample = 0 # 该对象一共有几个comment_type

        for attribute in attributes:
            if classification[attribute] != 0:
                attribute_counts[attribute] += 1
                comment_type_count_of_sample +=1

        object_counts[comment_type_count_of_sample] += 1

    # 输出属性的出现次数和对象的属性数量分布
    if output_file_path:
        with open(output_file_path, 'a') as f:
            f.write("Classification Counts:\n")
            for attribute, count in attribute_counts.items():
                f.write(f"{attribute}: {count}\n")

            f.write("\nSample Classification Counts:\n")
            for num_attributes, count in object_counts.items():
                f.write(f"Num of samples which has {num_attributes} comment types: {count}\n")
    else:
        print("Classification Counts:")
        for attribute, count in attribute_counts.items():
            print(f"{attribute}: {count}")

        print("\nSample Classification Counts:")
        for num_attributes, count in object_counts.items():
            print(f"Num of samples which has {num_attributes} comment types: {count}\n")
    return

def account_clone_detection_results(clone_detected_file_path,output_file_path):
    json_obj_list = read_json_file(clone_detected_file_path)


    # 统计属性出现次数和对象的属性数量分布
    attribute_counts = {"textual": 0, "lexical": 0, "syntactic": 0, "semantic": 0}
    object_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for json_obj in json_obj_list:
        classification = json_obj['clone_type']
        attributes = classification.keys()
        comment_type_count_of_sample = 0 # 该对象一共有几个clone type

        for attribute in attributes:
            if classification[attribute] != 0:
                attribute_counts[attribute] += 1
                comment_type_count_of_sample +=1

        object_counts[comment_type_count_of_sample] += 1

    # 输出属性的出现次数和对象的属性数量分布
    if output_file_path:
        with open(output_file_path, 'a') as f:
            f.write("\nClone Detection Counts:\n")
            for attribute, count in attribute_counts.items():
                f.write(f"{attribute}: {count}\n")

            f.write("\nSample Clone Counts:\n")
            for num_attributes, count in object_counts.items():
                f.write(f"Num of samples which has {num_attributes} clone types: {count}\n")
    else:
        print("Clone Detection Counts:")
        for attribute, count in attribute_counts.items():
            print(f"{attribute}: {count}")

        print("\nSample Clone Counts:")
        for num_attributes, count in object_counts.items():
            print(f"Num of samples which has {num_attributes} clone types: {count}")
    return







if __name__ == "__main__":
    pass
    # file_name = 'test'
    # input_file_dir = '../data'
    # output_file_dir = '../data'
    #
    # train_data_path = f'{input_file_dir}/{file_name}.jsonl'
    # classified_file_path = f'{output_file_dir}/{file_name}_classified.jsonl'
    # clone_detected_file_path = f'{output_file_dir}/{file_name}_detected.jsonl'
    #
    # account_classification_results(clone_detected_file_path)
    # account_sample_num(clone_detected_file_path)
