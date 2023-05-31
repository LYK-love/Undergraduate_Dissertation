import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
# from focal_loss.focal_loss import FocalLoss

# 定义神经网络模型
input_size = 5 * 4 # 5*4 matrix
hidden_size = 10
output_size = 1

num_epochs = 10


threshold = 0.6

batch_size = 32
num_class = 1

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
            # nn.ReLU()
        )

    def forward(self, x):
        # Reshape the input tensor to flatten it into a 1D tensor
        x = x.view(x.size(0), -1)

        hidden_output = self.hidden(x)
        output = self.out(hidden_output)
        # output = (output > threshold).float()

        return output.squeeze(-1)


class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_data = self.inputs[index]
        label = self.labels[index]
        # 在这里进行数据的预处理或转换
        return input_data, label

# 训练模型
def train_model(model, train_data_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_data_loader:
            # 梯度清零

            optimizer.zero_grad()

            # 前向计算和损失计算
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和参数更新
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_data_loader)
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# class TestFocalLoss():
#     def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
#         super(MultiFocalLoss, self).__init__()
#         self.gamma = gamma
#     def forward(self, logit, target):
#         alpha = self.alpha.to(logit.device)
#         prob = F.softmax(logit, dim=1)
#         ori_shp = target.shape
#         target = target.view(-1, 1)
#         prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan        logpt = torch.log(prob)
#         alpha_weight = alpha[target.squeeze().long()]
#         loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt
#         if self.reduction == 'mean':
#             loss = loss.mean()
#         return loss





def evaluate_model(model, test_data_loader):
    model.eval()
    with torch.no_grad():
        tp = 0  # True Positive
        tn = 0  # True Negative
        fp = 0  # False Positive
        fn = 0  # False Negative

        for data, label in test_data_loader:
            outputs = model(data)
            predicted_labels = torch.round(outputs)

            tp += ((predicted_labels == 1) & (label == 1)).sum().item()
            tn += ((predicted_labels == 0) & (label == 0)).sum().item()
            fp += ((predicted_labels == 1) & (label == 0)).sum().item()
            fn += ((predicted_labels == 0) & (label == 1)).sum().item()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        hamming = (fp + fn) / (tp + tn + fp + fn)

        # Convert tp, tn, fp, fn to binary true labels and predictions
        true_labels = torch.cat([torch.ones(tp + fn), torch.zeros(tn + fp)])
        pred_labels = torch.cat([torch.ones(tp + fp), torch.zeros(tn + fn)])

        roc_auc = roc_auc_score(true_labels, pred_labels)
        pres, recs, thres = precision_recall_curve(true_labels, pred_labels)
        prc_auc = auc(recs, pres)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "prc_auc": prc_auc,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn
        }


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Save model end. Model path: {path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path))

def extract_and_convert_to_tensor(json_list, attributes):
    tensors = []
    for json_obj in json_list:
        values = [json_obj['classification'][attr] for attr in attributes]
        tensor = torch.tensor(values, dtype=torch.float)
        tensors.append(tensor)
    return torch.stack(tensors)

def get_matrix_tensor(json_obj_list):
    '''
    5 * 4
    Args:
        json_obj_list:

    Returns:

    '''
    tensors = []
    for json_obj in json_obj_list:
        tensor = torch.zeros((5, 4), dtype=torch.float)

        classification = json_obj["classification"]
        clone_type = json_obj["clone_type"]

        for i, class_attr in enumerate(classification.keys()):
            for j, clone_attr in enumerate(clone_type.keys()):
                tensor[i, j] += classification[class_attr] * clone_type[clone_attr]
        # print(tensor)
        tensors.append(tensor)


    # tensors = get_matrix_tensor_under_control(json_obj_list, use_5D=False, use_4D=True)
    return tensors


def get_matrix_tensor_under_control(json_obj_list, use_5D, use_4D):
    '''
    用于对比实验

    Args:
        json_obj_list:
        use_5D: 使用子注释克隆分类
        use_4D: 使用子注释克隆检测

    Returns:

    '''

    tensors = []
    for json_obj in json_obj_list:
        tensor = torch.zeros((5, 4), dtype=torch.float)

        classification = json_obj["classification"] # 5D
        clone_type = json_obj["clone_type"]# 4D

        for i, class_attr in enumerate(classification.keys()):
            for j, clone_attr in enumerate(clone_type.keys()):
                if use_5D:
                    tensor[i, j] += classification[class_attr] * ( clone_type[clone_attr] * 0.8)
                elif use_4D:
                    tensor[i, j] += (classification[class_attr] * 0.8) * clone_type[clone_attr]
                else:
                    tensor[i, j] += classification[class_attr] * clone_type[clone_attr]
        tensors.append(tensor)
    return tensors


def get_target_tensor(json_obj_list):
    '''

    Args:
        json_obj_list:

    Returns:

    '''
    def extract_labels(json_obj_list):
        labels = []
        for obj in json_obj_list:
            label_value = obj['label']
            if label_value == True:
                label_value = 1.0
            else:
                label_value = 0.0
            labels.append(label_value)
        return labels

    target_labels = extract_labels(json_obj_list)
    return torch.tensor(target_labels, dtype=torch.float)

def train_and_save(input_file_path, model_file_path):

    # 读取json文件中的json对象, 作为nn的输入
    with open(input_file_path) as f:
        data = [json.loads(line) for line in f]

    # attributes_to_extract = ['what', 'why', 'how-it-is-done', 'how-to-use', 'property']
    # clone_type_attributes_to_extract = ["textual", "lexical", "syntactic", "semantic"]

    # 提取classification和clone_type属性并转换为张量
    input_tensor = get_matrix_tensor(data)

    # 训练集中相似内容的JSON对象的"label"属性值组成的列表
    target_tensor = get_target_tensor(data)


    model = NeuralNetwork(input_size, hidden_size, output_size)

    dataset = CustomDataset(input_tensor, target_tensor)

    # 数据加载器
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 定义损失函数和优化器
    # weight = torch.FloatTensor(batch_size, 1).random_(1) # 根据实际情况设置类别权重

    # weight = torch.FloatTensor([10.0]) # 根据实际情况设置类别权重
    criterion = nn.BCELoss()
    # Withoout class weights
    # weights = torch.FloatTensor([2, 3.2, 0.7])
    # criterion = FocalLoss(gamma=0.7, weights=weights)

    # criterion = FocalLoss(gamma=0.7)
    # criterion = MultiFocalLoss(num_class=num_class, gamma=2.0, reduction='mean')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型


    print("Training Begin.")
    train_model(model, data_loader, criterion, optimizer, num_epochs)
    print("Training End.")


    # 评估模型
    print("Evaluate model....")
    evaluation_result = evaluate_model(model, data_loader)
    print_evaluation_result(evaluation_result)

    # 存储模型
    save_model(model, model_file_path)
    print(f"Model saved to {model_file_path}")





def infer(test_dataset_path, model_file_path, evaluation_file_path):

    # 加载模型
    loaded_model = NeuralNetwork(input_size, hidden_size, output_size)
    load_model(loaded_model, model_file_path)

    # 使用加载的模型进行预测
    with open(test_dataset_path) as f:
        new_data = [json.loads(line) for line in f]

    new_input_tensor = get_matrix_tensor(new_data)  # 待预测的新数据转换后的张量
    new_target_tensor = get_target_tensor(new_data)

    test_dataset = CustomDataset(new_input_tensor, new_target_tensor)

    # 数据加载器
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print("Infer model....")
    infer_result = evaluate_model(loaded_model, test_data_loader)

    print_evaluation_result(infer_result)
    save_evaluation_result(infer_result, evaluation_file_path)

def save_evaluation_result(evaluation_result, evaluation_file_path):
    with open( evaluation_file_path, 'a') as f:
        f.write(datetime.now().strftime("%m/%d/%Y %H:%M:%S:"))
        f.write(json.dumps(evaluation_result) + '\n')

    print(f"Evaluation result saved to {evaluation_file_path}")
def print_evaluation_result(evaluation_result):
    def print_dictionary(dictionary):
        for key, value in dictionary.items():
            print(f"{key}: {value}")
    print_dictionary(evaluation_result)

if __name__ == "__main__":
    pass

#
#
#
# import json
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.embedding = nn.Embedding(2, 5)  # 输入是 0 或 1，输出是 5 维向量
#         self.fc1 = nn.Linear(5, 10)
#         self.fc2 = nn.Linear(10, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         embedded = self.embedding(x)
#         embedded = embedded.view(embedded.size(0), -1)  # 展平张量
#         x = self.fc1(embedded)
#         x = self.sigmoid(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         return x
#
# def preprocess_data(vectors):
#     encoded_vectors = []
#     for vector in vectors:
#         encoded_vector = [torch.tensor(value) for value in vector]
#         encoded_vectors.append(torch.stack(encoded_vector))
#     return torch.stack(encoded_vectors)
#
# def train_model(model, inputs, labels, criterion, optimizer, num_epochs):
#     model.train()
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
#
# def evaluate_model(model, inputs, labels):
#     model.eval()
#     with torch.no_grad():
#         outputs = model(inputs)
#         predicted_labels = (outputs >= 0.5).float()
#         accuracy = (predicted_labels == labels).float().mean()
#         precision = (predicted_labels * labels).sum() / predicted_labels.sum()
#         recall = (predicted_labels * labels).sum() / labels.sum()
#         f1 = 2 * (precision * recall) / (precision + recall)
#         hamming_loss = (predicted_labels != labels).float().mean()
#
#     print(f"Accuracy: {accuracy.item():.4f}")
#     print(f"Precision: {precision.item():.4f}")
#     print(f"Recall: {recall.item():.4f}")
#     print(f"F1 Score: {f1.item():.4f}")
#     print(f"Hamming Loss: {hamming_loss.item():.4f}")
#
# with open('./data/train_data_merged.jsonl') as f:
#     data = [json.loads(line) for line in f]
#
# attributes_to_extract = ['what', 'why', 'how-it-is-done', 'how-to-use', 'property']
#
# inputs = extract_and_convert_to_tensor(data, attributes_to_extract)
#
#
# target_labels = extract_labels(data)# 训练集中相似内容的JSON对象的"label"属性值组成的列表
#
# labels = torch.tensor(target_labels, dtype=torch.float32).view(-1, 1)
#
# # 创建模型实例
# model = Net()
#
# # 定义损失函数和优化器
# criterion = nn.BCELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
#
#
#
#
# # 训练模型
# num_epochs = 10
# train_model(model, inputs, labels, criterion, optimizer, num_epochs)
#
# # 评估模型
# evaluate_model(model, inputs, labels)
#
#
#
