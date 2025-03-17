import os
import argparse
import json
from tqdm import tqdm
from pdb import set_trace as pds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default="./logs_0.85/submissions/vqav2-test-submission-2025-03-13-15-11-08.json")
    parser.add_argument('--dst', type=str, default="answers_upload/submission2.json")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    src = os.path.join(args.src)
    test_split_ids = 'scripts/convert/vqav2_test_split_question_ids.json'
    dst = os.path.join(args.dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    with open(src, 'r') as file:
        results = json.load(file)
    
    with open(test_split_ids, 'r') as file:
        question_ids = json.load(file)

    results = {x['question_id']: x['answer'] for x in results}
    print(f'total results: {len(results)}, total split: {len(question_ids)}')

    all_answers = []

    for question_id in tqdm(question_ids):
        if question_id not in results:
            all_answers.append({
                'question_id': question_id,
                'answer': ''
            })
        else:
            all_answers.append({
                'question_id': question_id,
                'answer': results[question_id]
            })

    with open(dst, 'w') as f:
        json.dump(all_answers, open(dst, 'w'))
