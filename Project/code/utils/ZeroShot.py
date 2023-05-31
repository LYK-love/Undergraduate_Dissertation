
import json
from transformers import pipeline

def add_classification_property_using_model(json_obj_list):
    model_name = "facebook/bart-large-mnli"
    pipe = pipeline(model=model_name, cache_dir="models")


    print(f"Begin Classification. Use zero-shot model: {model_name}")
    texts = []
    for json_obj in json_obj_list:
        old_comment = json_obj["src_desc"]
        new_comment = json_obj["dst_desc"]

        old_code = json_obj["src_method"]
        new_code = json_obj["dst_method"]

        sample_for_classification = "Comment: " + old_comment + "\n" + "Code: " + old_code
        texts.append(sample_for_classification)

    result = pipe(texts,
                  candidate_labels=["what", "why", 'how-it-is-done', 'how-to-use', 'property']
                  )

    for json_obj, sample_res in zip(json_obj_list, result):

        # sample_res:
        # {'sequence': 'I hacdscdve a prcddscdoblem with my acdcds that needs to be resolved asap!',
        #       'labels': [, 'what', 'why', "how-it-is-done", 'how-to-use', 'property'],
        #       'scores': [0.47700586915016174, 0.26049768924713135, 0.20162972807884216, 0.06086677312850952]
        #       }
        scores = sample_res['scores'] # list, len = 5. [0.47700586915016174, 0.26049768924713135, 0.20162972807884216, 0.06086677312850952]

        # classification_value_list = [1 if num > 0.15 else 0 for num in scores]

        # property > "how-it-is-done" > what > 'how-to-use' > why
        classification_value_list = []
        for idx, score in enumerate(scores):
            classification_value = 0
            if idx == 0: # what
                if score > 0.2:
                    classification_value = 1
            elif idx == 1: # 'why'
                if score > 0.25:
                    classification_value = 1
            elif idx == 2: # "how-it-is-done"
                if score > 0.08:
                    classification_value = 1
            elif idx == 3: # 'how-to-use'
                if score > 0.15:
                    classification_value = 1
            elif idx == 4: # 'property'
                if score > 0.01:
                    classification_value = 1
            classification_value_list.append(classification_value)



        classification_key_list = ['what', 'why', 'how-it-is-done', 'how-to-use', 'property']

        classification = {key: value for key, value in zip(classification_key_list, classification_value_list)}

        # 不允许全1或全0的情况
        if all(value == 0 for value in classification.values()):
            classification['property'] = 1
        elif all(value == 1 for value in classification.values()):
            print("All One!")
            classification['what'] = 0
            classification['why'] = 0
            classification['how-it-is-done'] = 1,
            classification['how-to-use'] = 0
            classification['property'] = 1
        json_obj["classification"] = classification

        # print(scores)
        # print(json_obj["classification"])

    return json_obj_list