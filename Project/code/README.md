# Intro
This repository is an implementation of my undergraduate dissertation: 

项目目录结构如下:

```
.
├── Project
│   ├── code
│   ├── results
│   └── subjects
└── Thesis
```



* Thesis：论文的PDF和源文件（latex文件及所有辅助图片文件等，必须可编译）
* Project:
  * code：存放毕设项目代码，需包含一个readme。readme中需包括

    - requirements: 说明项目运行依赖的环境、库版本等;

    - setup: 需要做的配置；

    - input：输入的内容和格式；

    - usage：运行的方法；

    - output：输出的格式以及如何解读。
  * subjects：存放实验中用到的数据集.
  * results：存放实验结果.

# Requirements

python=3.8

Unbuntu 20.04, x86_84 with cuda 11.8



`requirements.txt`:

```txt
certifi==2022.9.24
charset-normalizer==2.1.1
click==8.1.3
filelock==3.11.0
fuzzywuzzy==0.18.0
idna==3.4
javalang==0.13.0
nltk==3.7
numpy==1.24.2
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
packaging==23.0
PyYAML==6.0
regex==2022.10.31
requests==2.28.1
scikit-learn==1.1.3
scipy==1.10.1
six==1.16.0
threadpoolctl==3.1.0
tokenizers==0.13.2
torch==1.13.1
tqdm==4.63.1
transformers==4.27.4
typing_extensions==4.5.0
urllib3==1.26.13
```



# setup

需要做的配置:

1. 下载并解压数据集: 

   1. 下载数据集: https://box.nju.edu.cn/f/c992235d3b3d41f8b664/?dl=1
   2. 解压到`subjects/`目录, 重命名为`data`.

   数据集的格式参见[下文](# 数据集).

2. 在下载数据集后, 你不需要更改其他的设置, 就可以运行该项目的步骤**3~5.** (参见[项目的运行](#usage))

   2. 如果你想要运行步骤2, 那么需要安装并启动core-nlp-server:

   1. install: https://stanfordnlp.github.io/CoreNLP/download.html

      * 从官方教程可以看到, 运行core-nlp-server还需要安装**jdk8**.

        ```
        sudo apt-get install upgrade
        sudo apt-get install openjdk-8-jdk
        ```

   2. Start core-nlp-server: https://stanfordnlp.github.io/CoreNLP/corenlp-server.html#getting-started

      Stanford CoreNLP ships with a built-in server, which requires only the CoreNLP dependencies. To run this server, simply run:

      ```
      # Run the server using all jars in the current directory (e.g., the CoreNLP home directory)
      java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
      ```

   3. 启动core-nlp-server后, 就可以运行步骤2了.



## 项目变量

所有变量都在`run.py`中, 它们指定了各种输入输出文件的位置. 如果你想要自定义实验的步骤, 那么可以更改它们.

默认情况下**不需要更改任何变量**. 详情请见: [项目的运行](#usage)

```python
    # "Project"文件夹的路径
    project_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # "Project/code"文件夹的路径
    code_dir_path = os.path.dirname(os.path.abspath(__file__))

    # "Project/subjects"文件夹的路径
    subjects_dir_path = f"{project_dir_path}/subjects"

    # "Project/results"文件夹的路径
    results_dir_path = f"{project_dir_path}/results"

    print(code_dir_path)

    raw_dataset_dir_path = f'{subjects_dir_path}/data/raw_dataset'
    cleaned_dataset_dir_path = f'{subjects_dir_path}/data/cleaned_dataset'  # 清洗后的数据集
    middle_dataset_dir_path = f'{code_dir_path}/data_process'  # 清洗时产生的中间结果
    processed_middle_dataset_dir_path = f'{subjects_dir_path}/data/middle_processed_dataset'  # 进一步处理后的中间产物数据集
    processed_dataset_dir_path = f'{subjects_dir_path}/data/processed_dataset'  # 数据集处理完后存放的目录

    # 待分析的数据集们所在的目录
    dataset_dir_to_analyze = processed_dataset_dir_path
    # 每个数据集都对应一个同名的分析结果文件(例如, `train.json`对应的分析结果是`train.json.txt`), 存放在分析结果目录:
    analysis_results_dir_path = f'{results_dir_path}/analysis_results'

    training_dataset_path = f'{processed_dataset_dir_path}/train.jsonl'
    test_dataset_path = f'{processed_dataset_dir_path}/test.jsonl'

    # 训练后保存的模型:
    model_file_name = "model.pth"
    output_model_file_dir = f'{results_dir_path}/models'
    output_model_file_path = f'{output_model_file_dir}/{model_file_name}'

    # 评估时输入的模型:
    input_model_file_path_for_infer = output_model_file_path

    # 评估结果保存位置, 默认保存在'<project>/eval.log'
    evaluation_file_path = f'{results_dir_path}/eval.log'
```



## 数据集

我提供的数据集的目录结构为:

```
.
├── cleaned_dataset
├── cup2_dataset
├── cup2_updater_dataset
├── label_res
├── label_view
├── labeled_dataset
├── middle_processed_dataset
├── processed_dataset
└── raw_dataset
```

* `label_res`: 存放了人工分类的结果. 在步骤2处理时, 会读取该目录下的csv文件, 对数据集进行分类.
* `raw_dataset`: 存放了步骤1所需的原始数据集. 
  * 你也可以按照下面的[教程](#下载其他数据集)下载一份自己的数据集, 然后替换掉我给的这份. 但是我的人工分类结果是针对我给的这份数据集的，因此如果想使用新的数据集，必须自己提供一份人工分类文件,
* `cleaned_dataset`: 存放了步骤1处理后生成的数据集, 即数据清洗后的数据集.
* `middle_processed_dataset`: 存放了步骤2处理时生成的数据集, 这些数据集会在步骤2中被继续利用.
* `processed_dataset`: 存放了步骤2处理后生成的数据集, 也就是模型的**测试集和训练集**:
  * Training set: `processed_dataset/train.jsonl`
  * Test set: `processed_dataset/test.jsonl`

### 下载其他数据集

本论文数据集来自[CUP2](https://github.com/Tbabm/CUP2), 你也可以从CUP2下载数据集来替换掉我使用的数据集`raw_dataset`.

* Download the dataset from [here](https://drive.google.com/drive/folders/1FKhZTQzkj-QpTdPE9f_L9Gn_pFP_EdBi?usp=sharing)





# input



# usage

## Quick Start

最简单的方法: 在完成[setup](#setup)后,

1. cd to code dir:

   ```
   cd <project>/Project/code
   ```

2. run `run.py`:

   ```
   python run.py
   ```

***

本项目分为五个步骤, `python run.py`会顺序执行这五个步骤. 你也可以单独执行这些步骤.

## Step1 clean dataset

清洗数据集:

```
python run.py --clean_dataset
```

该方法会将`raw_dataset_dir_path`下的文件作为原始数据集, 进行清洗, 结果输出到`cleaned_dataset_dir_path`

## Step2 process dataset

进一步处理数据集, 用于后续的模型训练:

```
python run.py --process_dataset
```

该方法会将`cleaned_dataset_dir_path`下的文件作为输入, 进行: 注释分类, 子注释克隆检测, 样本合并和人工数据标注.

结果输出到`processed_dataset_dir_path`.

* 人工数据标注: 人工分类的结果文件位于`label_res`目录下, 在这一步骤会读取这些文件.



## Step3 analyze dataset

统计数据集的特征:

```
python run.py --analyze_dataset
```

该方法会将`processed_dataset_dir_path`下的文件作为输入, 统计该目录下每个数据集的特征, 结果输出到`analysis_results_dir_path`.

## Step4 train

使用数据集中的**训练集**, 训练并存储该模型:

```
python run.py --train
```

该方法会将`training_dataset_path`作为训练集, 训练模型, 模型会存储到`output_model_file_path`

## Step4 infer

给定数据集中的**测试集**和模型, 评估该模型:

```sh
python run.py --infer
```

该方法会将`test_dataset_path`作为测试集, 将`input_model_file_path_for_infer`作为要评估的模型, 评估结果存储到`evaluation_file_path`

## Notes

* 在下载了原始数据集的情况下, 你可以依次运行1~5来得到结果.

* 如果你不想下载原始数据集, 那么我提供了一份“第2步处理”后的数据集, 你可以运行3~5 或者 4~5 来在该数据集上训练模型, 得到结果.

* 如果你不想训练模型, 仅仅想进行模型的评估(Infer), 那么我提供了一个训练好的模型, 你可以运行5来得到模型的评估结果.
* 在上面的三种情况下, 你**不需要更改任何的路径变量**, 只需采用我默认的配置. 当然你也可以按自己的需要更改. 例如, 如果你想要测试不经过清洗的数据集和清洗后的数据集(即是否执行步骤1)对模型训练的效果, 只需更改步骤2的输入数据集目录为`raw_dataset_dir_path`, 而非`cleaned_dataset_dir_path`即可.

# output

项目的输出存储在`<project>/Project/results/`中:

```
.
├── analysis_results
│   ├── file_path
│   ├── full_valid.jsonl.txt
│   ├── test.jsonl.txt
│   └── train.jsonl.txt
├── eval.log
└── models
    ├── model.pth
    └── readme.md
```

* `analysis_results/`: 存储了对每个数据集文件的分析, 数据集`XX.json` 对应分析文件`XX.json.txt`.
* `eval.log`: 存储了模型的评估结果.
* `models`: 存储了训练得到的模型. 每次训练默认会把模型存在`models/model.pth`. 
  * 如果你要自己训练并生成模型, 可以更改模型路径.

# 老师的指示

请在https://gitee.com/上创建毕设项目仓库。

项目名称：2023\_姓名\_本科毕设_论文名

访问权限：private

完成上述步骤后，分享给mxpan(mxpan@outlook.com)。请在仓库成员管理中选择“直接添加”而不是“链接邀请”。