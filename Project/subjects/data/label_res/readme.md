# Task
对每个`processed_dataset/{filename}.jsonl`文件进行人工打标签, 结果存放在`label_res/{filename}.csv`.

`filename` in {fill_valid, test, train}.

`processed_dataset/{filename}.jsonl`的每一行都是一个json对象. 其格式如下:
``` json
{ 
"idx": 0, 
"sample_id": 1033751,  
"src_method": "XXX", 
"dst_method": "XXX", 
"src_desc": "XXX", 
"dst_desc": "XXXX",
"classification": 
    {
    "what": 0, 
    "why": 0, 
    "how-it-is-done": 1, 
    "how-to-use": 1, 
    "property": 1
    }, 
"clone_type": 
    {
    "textual": 1, 
    "lexical": 0, 
    "syntactic": 0, 
    "semantic": 0
    }
... // 别的属性, 和打标签无关
}
```

你需要:
1. 创建对应的`label_res/{filename}.csv`, 写入header, header是`idx,sample_id,label`三个字段. 
2. 接着根据已设定的原则对每个json对象打标签, 每个json对象的打标签结果是`label_res/{filename}.csv`的一行.
```csv
idx,sample_id,label
0,5444123,False
0,3330620,False
0,2230530,False
```
在这个例子中, `idx=0`, `sample_id=5444123`的json对象被标记为False.

# Label Principle
打标签的过程很简单, 只需关注每个样本(json对象)的"classification"和"clone_type"属性, 对样本进行打分. 每个样本的threshold(阈值)为5. 当且仅当样本的score(分数)超过阈值时, 该样本被标记为True.
```phthon
score = 0
threshold = 5

# compute score...
label = False

if score > threshold:
    label = True
```

打分规则如下:
1. 如果样本的"property"为True(即classification"中"property"的属性值为1, 下同理), 且“lexical”为True, 则score += 2.
2. 如果样本的"how-it-is-done"为True, 且“textual”为True, 则score += 1.
3. 如果样本的"what"为True, 且“syntactic”为True, 则score += 1.
4. 如果样本的"how-to-use"或"why"为True, 且“semantic”为True, 则score += 1.
5. 在进行上述加分后,如果在score < threshold 的情况下, 实验者依然认为"src_desc"和"dst_desc"的内容具有注释克隆的关系, 则score += 1.


标注人:
1. 陆昱宽
2. 汪喆









