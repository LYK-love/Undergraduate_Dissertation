import argparse
import os
import sys

# "Project"文件夹的路径
project_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 项目根文件夹的路径
root_dir_path = os.path.dirname(project_dir_path)

sys.path.append(project_dir_path)
sys.path.append(root_dir_path)

print(f"Module Searching Path: {sys.path}")

from nn import train_and_save, infer
from utils.FIle_process import process_sentence_and_save, merge_samples_and_save, modify_and_clean_dataset, \
    add_label, get_line_cnt, get_dataset_ratio, process_sentence, aggregate_comments, \
    merge_samples_and_save_using_data, change_label_use_heuristic_using_path, save_jsonl_file, rewrite_jsonl_file

from utils.Statistics import analysis


def clean_dataset(raw_dataset_dir_path, middle_dataset_dir_path, cleaned_dataset_dir_path):
    original_dataset_dir = f'{raw_dataset_dir_path}/cup2_dataset'  # 原始的CUP2_dataset数据集
    original_update_dataset_dir = f'{raw_dataset_dir_path}/cup2_updater_dataset'  # 原始的CUP2_updater_dataset数据集

    modify_and_clean_dataset(original_dataset_dir, original_update_dataset_dir, cleaned_dataset_dir_path,
                             middle_dataset_dir_path, middle_dataset_dir_path)
    return


def process_dataset(input_dataset_dir_path, output_middle_dataset_dir_path, output_merged_dataset_dir_path, output_labeled_dataset_dir_path, manually_label_res_csv_dir_path, use_manually_label, is_save = False):
    '''
    数据处理和人工标注部分.该方法在最后还会调整训练集和测试集的比例.
    数据处理包括: 子注释分类, 子注释克隆检测, 样本合并.

    Args:
        input_dataset_dir_path: 由于采用了数据清洗, 这里是清洗后数据集的目录, 应该只存在三个文件: ['train', 'full_valid', 'test']. 这是因为数据清洗的源目录是"{raw_dataset_dir_path}/cup2_dataset", 里面只有 ['train', 'full_valid', 'test']
        output_middle_dataset_dir_path: 数据处理过程中产生的中间产物(子注释分类,子注释克隆检测)数据集.
        output_merged_dataset_dir_path: 样本合并(代码-子注释变更样本 合并为 代码-注释变更样本)后的数据集.
        output_labeled_dataset_dir_path: 对合并后的样本进行人工标注后的数据集.
        manually_label_res_csv_dir_path: 人工标注的结果被保存到该目录下, 该函数会读取该目录下的文件, 用其中的结果来给数据集添加label.
        use_manually_label: 是否使用人工标注. 如果为False, 则不会使用{manually_label_res_csv_dir_path}, 而会使用数据集中默认的label(这是CUP2中的label, 用于表明该样本是否具有过时注释, 这个label对注释克隆的意义不大.)
        is_save: 是否存储中间产物, 即{output_merged_dataset_dir_path}. 但是合并后的样本无论如何都会被存储.

    Returns:

    '''

    file_name_list = ['train',  'full_valid', 'test']  # 没有‘valid’
    # file_name_list = ['train']  # 没有‘valid’


    for file_name in file_name_list:
        dataset_path = f'{input_dataset_dir_path}/{file_name}.jsonl'
        classified_file_path = f'{output_middle_dataset_dir_path}/{file_name}_classified.jsonl'
        clone_detected_file_path = f'{output_middle_dataset_dir_path}/{file_name}_detected.jsonl'
        merged_file_path = f'{output_merged_dataset_dir_path}/{file_name}.jsonl'

        manually_label_res_csv_file_path = f'{manually_label_res_csv_dir_path}/{file_name}.csv' # 人工分类的结果被存储为csv文件, 位于{manually_label_res_csv_dir_path}下.
        labeled_file_path = f"{output_labeled_dataset_dir_path}/{file_name}.jsonl" #人工标注生成的数据集的路径

        if is_save:
            # 利用清洗后的数据集作为输入, 来进一步处理
            process_sentence_and_save(dataset_path, classified_file_path, clone_detected_file_path)
            # 接下来, 将sentence为单元的注释合并成comment注释
            merge_samples_and_save(clone_detected_file_path, merged_file_path)


        else:
            processed_data = process_sentence(dataset_path)
            merge_samples_and_save_using_data(processed_data, merged_file_path)

        # 现在, 数据集大小已经确定了. 因为注释已经被全部合成了. 后面的人工标注也不会改变数据集大小.

        # 对于合并的样本，按照定义好的注释克隆规则进行人工标注.

        # 人工标注的结果存放在{manually_label_res_csv_file_path}, 该函数会读取这份文件, 对数据集的样本进行标注。
        # 标注后的数据被存在{labeled_file_path}.
        add_label(merged_file_path, labeled_file_path, manually_label_res_csv_file_path)



    # 事实上, 这个选择是多此一举的, 因为output_processed_dataset_dir_path和output_labeled_dataset_dir_path下面的数据集的大小是相同的.
    if use_manually_label:
        # 我们将人工标注后的数据作为数据集
        training_dataset_path = f'{output_labeled_dataset_dir_path}/train.jsonl'
        test_dataset_path = f'{output_labeled_dataset_dir_path}/test.jsonl'
    else:
        # 我们没有使用人工标注(在这种情况下,使用CUP2数据集原本的标注)作为数据集
        training_dataset_path = f'{output_merged_dataset_dir_path}/train.jsonl'
        test_dataset_path = f'{output_merged_dataset_dir_path}/test.jsonl'

    modify_dataset_ratio(training_dataset_path,test_dataset_path,0.8)

    return

def modify_dataset_ratio(training_dataset_path, test_dataset_path, ratio = 0.8):
    '''
    查看训练集和测试集的比例,如果训练集没有占整个数据集的{ratio}(80% by default), 则删减训练集直到比例达到{ratio}
    Args:
        training_dataset_path:
        test_dataset_path:
        ratio:

    Returns:

    '''
    training_dataset_size = get_line_cnt(training_dataset_path)
    test_dataset_size = get_line_cnt(test_dataset_path)
    dataset_ratio = get_dataset_ratio(training_dataset_path, test_dataset_path)

    print(f"Num of samples in training dataset: {training_dataset_size}")
    print(f"Num of samples in test dataset: {test_dataset_size}")
    print(f"Dataset ratio(should be close to 0.8): {dataset_ratio}")

    if dataset_ratio > 0.82 or dataset_ratio < 0.78:
        print(f"Fix dataset ratio to 0.8, we do this by cutting training dataset.")
        correct_training_data_size = test_dataset_size * 4
        rewrite_jsonl_file(training_dataset_path, training_dataset_path, correct_training_data_size)

        # Update training dataset size and ratio
        training_dataset_size = get_line_cnt(training_dataset_path)
        dataset_ratio = get_dataset_ratio(training_dataset_path, test_dataset_path)

        print("Now the ratio is fixed to 0.8.")
        print(f"Num of samples in training dataset: {training_dataset_size}")
        print(f"Num of samples in test dataset: {test_dataset_size}")
        print(f"Dataset ratio(should be  0.8): {dataset_ratio}")

def analyze_all_dataset(input_dataset_dir_path, analysis_results_dir_path):
    '''
    给定一个目录, 该目录下有多个数据集. 对每个数据集进行统计.
    Args:
        input_dataset_dir_path:

    Returns:

    '''
    # 遍历目录下的所有文件
    print(f"Analyze datasets in dir {input_dataset_dir_path}")

    for root, dirs, files in os.walk(input_dataset_dir_path):
        for file in files:
            file_path = os.path.join(root, file)  # 目录下的文件的路径, 这里就是各个数据集的路径

            # 判断文件是否存在
            if os.path.isfile(file_path):
                file_name = os.path.splitext(file)[0]
                analysis_result_file_path = f"{analysis_results_dir_path}/{file}.txt"
                analysis(file_path, analysis_result_file_path)  # 若存在该文件, 则统计它
    print(f"Analysis End. Save results in dir: {analysis_results_dir_path}")

    return

def no_arg(argv):
    is_no_arg = True
    for k,v in argv.items():
        if v == True:
            is_no_arg = False
            break
    return is_no_arg
def train_and_build_model(training_dataset_path, output_model_path):
    print(f"Create and train model. Use training dataset: {training_dataset_path}")
    train_and_save(training_dataset_path, output_model_path)
    return


def infer_model(test_dataset_path, input_model_path, evaluation_file_path):
    print(f"Infer model. Use test data set: {test_dataset_path}")
    infer(test_dataset_path, input_model_path, evaluation_file_path)


if __name__ == "__main__":
    '''
    '--clean' 是参数的名称，它表示一个命令行选项，以双横线开头。
action='store_true' 指定了参数的行为。store_true 表示当该选项在命令行中出现时，将该选项的值设为 True。如果未出现该选项，则默认为 False。这种行为常用于表示开关或标记，例如启用或禁用某个功能。
help='use edit seq for detection' 提供了关于参数的说明文本。当用户在命令行中使用 -h 或 --help 选项时，将显示这段说明文本。
    '''
    # sys.path.append("..")

    # Data definition:
    #####################################
    # 我们假设已经正确地下载了数据集( original_dataset_dir 和 original_update_dataset_dir), 并存放在了{raw_dataset_dir_path}目录


    #__file__ 是一个内置的全局变量，它包含当前脚本的文件名。os.path.abspath(__file__) 返回该文件的绝对路径。os.path.dirname() 函数获取该路径的父目录路径，因此 project_dir 将包含项目目录的路径



    # "Project/code"文件夹的路径
    code_dir_path = os.path.dirname(os.path.abspath(__file__))

    # "Project/subjects"文件夹的路径
    subjects_dir_path = f"{project_dir_path}/subjects"

    # "Project/subjects/data"文件夹的路径. 如果是debug阶段, 则使用"Project/subjects/data_for_test"
    data_dir_path = f'{subjects_dir_path}/data'

    # "Project/results"文件夹的路径
    results_dir_path = f"{project_dir_path}/results"

    print(code_dir_path)

    raw_dataset_dir_path = f'{data_dir_path}/raw_dataset'
    cleaned_dataset_dir_path = f'{data_dir_path}/cleaned_dataset'  # 清洗后的数据集
    middle_dataset_dir_path = f'{code_dir_path}/data_process'  # 清洗时产生的中间结果
    processed_middle_dataset_dir_path = f'{data_dir_path}/middle_processed_dataset'  # 进一步处理后的中间产物数据集
    merged_dataset_dir_path = f'{data_dir_path}/processed_dataset'  # 数据集处理后(经过了子注释分类和子注释克隆检测, 以及样本合并)存放的目录, 现在的每个数据集的样本都是合并后的“代码-注释变更”样本.
    labeled_dataset_dir_path = f'{data_dir_path}/labeled_dataset'  # 人工标注过后的数据集的目录

    manually_label_res_csv_dir_path = f'{data_dir_path}/label_res' # 数据集人工标注的结果是一份份csv文件, 存储在目录{manually_label_res_csv_dir_path{下

    save_middle_dataset = False  # 是否存储process_data中的分类和打标签之后产生的中间数据集. 注意, 聚合后的数据集和突变后的数据集无论如何都会被计算并存储
    use_manually_label = True # 使用人工标注的数据集作为模型的数据集. 默认情况下总是为True. 如果为False，则数据集不会进行人工标注，这样会采用数据集原本的label值(CUP2中的label值)


    if use_manually_label:
        dataset_dir_to_analyze = labeled_dataset_dir_path
        training_dataset_path = f'{labeled_dataset_dir_path}/train.jsonl'
        test_dataset_path = f'{labeled_dataset_dir_path}/test.jsonl'
    else:
        dataset_dir_to_analyze = merged_dataset_dir_path
        training_dataset_path = f'{merged_dataset_dir_path}/train.jsonl'
        test_dataset_path = f'{merged_dataset_dir_path}/test.jsonl'

    analysis_results_dir_path = f'{results_dir_path}/analysis_results'


    # 训练后保存的模型:
    model_file_name = "model.pth"
    output_model_file_dir = f'{results_dir_path}/models'
    output_model_file_path = f'{output_model_file_dir}/{model_file_name}'

    # 评估时输入的模型:
    input_model_file_path_for_infer = output_model_file_path

    # 评估结果保存位置, 默认保存在'<project>/eval.log'
    evaluation_file_path = f'{results_dir_path}/eval.log'
    #######################

    parser = argparse.ArgumentParser()

    parser.add_argument('--clean_dataset', action='store_true', help=f'Clean dataset')
    parser.add_argument('--process_dataset', action='store_true', help='Process dataset')
    parser.add_argument('--analyze_dataset', action='store_true', help='Analyze dataset')
    parser.add_argument('--train', action='store_true', help='Train and save model')
    parser.add_argument('--infer', action='store_true', help='Infer model')
    args = parser.parse_args()

    argv = vars(args)
    # 检测是否没有任何命令行参数
    if no_arg(argv):
        print("No args, execute all commands sequentially")
        # 首先清理数据集
        clean_dataset(raw_dataset_dir_path, middle_dataset_dir_path, cleaned_dataset_dir_path)

        # 接下来, 利用清洗后的数据集作为输入, 进行数据处理和人工标注.
        process_dataset(cleaned_dataset_dir_path, processed_middle_dataset_dir_path, merged_dataset_dir_path, labeled_dataset_dir_path, manually_label_res_csv_dir_path, use_manually_label, save_middle_dataset)

        # 统计处理后的数据集
        analyze_all_dataset(dataset_dir_to_analyze, analysis_results_dir_path)

        # Create and train model. Then save it.
        train_and_build_model(training_dataset_path, output_model_file_path)

        # Infer model
        infer_model(test_dataset_path, input_model_file_path_for_infer, evaluation_file_path)
    else:
        if args.clean_dataset:
            clean_dataset(raw_dataset_dir_path, middle_dataset_dir_path, cleaned_dataset_dir_path)
        if args.process_dataset:
            process_dataset(cleaned_dataset_dir_path, processed_middle_dataset_dir_path, merged_dataset_dir_path, labeled_dataset_dir_path, manually_label_res_csv_dir_path, use_manually_label, save_middle_dataset)
        if args.analyze_dataset:
            analyze_all_dataset(dataset_dir_to_analyze, analysis_results_dir_path)
        if args.train:
            train_and_build_model(training_dataset_path, output_model_file_path)
        if args.infer:
            infer_model(test_dataset_path, input_model_file_path_for_infer, evaluation_file_path)
        else:
            pass

