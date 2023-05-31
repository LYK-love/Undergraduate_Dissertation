import torch
from torch import cosine_similarity
from transformers import AutoTokenizer, AutoModel

def detect_semantic_clones(comment1, comment2, threshold=0.85):
    """
    Detects whether two Javadoc comments are semantic clones.
    """
    # # Load the pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    # Tokenize the comments and convert them to PyTorch tensors
    tokens1 = tokenizer(comment1, padding=True, truncation=True, return_tensors='pt')
    input_ids1 = tokens1['input_ids']
    attention_mask1 = tokens1['attention_mask']

    tokens2 = tokenizer(comment2, padding=True, truncation=True, return_tensors='pt')
    input_ids2 = tokens2['input_ids']
    attention_mask2 = tokens2['attention_mask']

    # Compute the BERT embeddings for the comments
    with torch.no_grad():
        outputs1 = model(input_ids1, attention_mask=attention_mask1)
        embeddings1 = outputs1.last_hidden_state[:, 0, :]

        outputs2 = model(input_ids2, attention_mask=attention_mask2)
        embeddings2 = outputs2.last_hidden_state[:, 0, :]


    # Compute the cosine similarity between the embeddings
    # embeddings.reshape(1, -1) 是将向量转换为一个大小为 (1, n) 的二维数组，
    # 其中 n 是向量的长度。这是为了适应 cosine_similarity 函数的输入要求，它需要接收两个二维数组作为输入。
    # 同样地，embeddings[1].reshape(1, -1) 将第二个向量转换为一个大小为 (1, n) 的二维数组。
    similarity = cosine_similarity(embeddings1.reshape(1, -1), embeddings2.reshape(1, -1)).item()

    # Return True if the cosine similarity is greater than a threshold value
    return similarity > threshold



if __name__ == '__main__':
    text1 = "This class is responsible for managing the current state of this node and ensuring only valid state transitions. Below we define the possible state transitions and how they are triggered."
    text2 = '''
    This class is used to serialize inbound requests or responses to outbound requests. 
    It basically just allows us to wrap a blocking queue so that we can have a mocked implementation which does not depend on system time. 
    See {@link org.apache.kafka.raft.internals.BlockingMessageQueue}.
    '''
    res = detect_semantic_clones(text1,text2)
    print(res)