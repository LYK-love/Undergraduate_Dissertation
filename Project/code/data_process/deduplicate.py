import argparse
from collections import defaultdict
from difflib import unified_diff
import json
from pathlib import Path


def deduplicate(original_dataset_dir, output_file_dir_path):
    original_dataset_dir = Path(original_dataset_dir)
    output_file_dir_path = Path(output_file_dir_path)
    output_file_path = output_file_dir_path / "id_delete.json"

    splits = ['train', 'full_valid', 'test'] #不存在'valid'
    # splits = ['train', 'test']

    count = 0
    code_diff_old_comment_dict = defaultdict(list)



    for s in splits:
        with open(original_dataset_dir / f'{s}.jsonl') as f:
            for l in f:
                count += 1
                obj = json.loads(l)
                old_code_lines = obj['src_method'].split('\n')
                new_code_lines = obj['dst_method'].split('\n')
                old_code_lines = [l.strip() for l in old_code_lines if l.strip() != '']
                new_code_lines = [l.strip() for l in new_code_lines if l.strip() != '']
                diff = '\n'.join(list(unified_diff(old_code_lines, new_code_lines, lineterm='', n=100))[3:])
                code_diff_old_comment_dict[diff + '\n' + obj['src_desc'].replace('\n', ' ')].append(
                    f"{obj['idx']}_{obj['sample_id']}: {int(obj['label'])}")

    print(len(code_diff_old_comment_dict), count)

    id_keep = []
    id_all = []
    iid = set()
    for v in code_diff_old_comment_dict.values():
        kid = ''
        klable = -1
        for s in v:
            id, label = s.split(': ')
            id_all.append(str(id))
            lable = int(label)
            if lable > klable:
                kid = id
                klable = lable
            if id.split('_')[1] in iid:
                kid = id
                klable = 2
        id_keep.append(str(kid))
        iid.add(kid.split('_')[1])
    id_delete = list(set(id_all) - set(id_keep))


    json.dump(id_delete, open(output_file_path, 'w'))
    print(f"DEDUPLICATION END! Save file to {output_file_path}")


if __name__ == "__main__":
    print("DUPLICATE START!")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_path', required=True, type=str, help='path of original data\'s dir, this file will make a "id_delete.json" file as output file!')
    parser.add_argument('--output_file_path', default='/root/projects/TestData/data_process/id_delete.json', type=str, help='path to output data, by default it will be a "id_delete.json" file in current dir! ')

    args = parser.parse_args()
    input_file_dir_path = Path(args.input_file_path)
    output_file_path = Path(args.output_file_path)
