import json
import re
import argparse

RELATION_TYPES = ['left', 'right', 'front', 'behind']
RELATION_PHRASES = {
    'left':   'to the left of',
    'right':  'to the right of',
    'front':  'in front of',
    'behind': 'behind'
}
INVERSE_REL = {
    'left':   'right',
    'right':  'left',
    'front':  'behind',
    'behind': 'front'
}

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def ablate_dataset(split: str, ablation_relation: str):
    in_path   = f'./data/CLEVR_{split}_qa.json'
    out_path  = f'./data/ablated_CLEVR/CLEVR_{split}_qa_ablate_{ablation_relation}.json'

    questions = read_json(in_path)
    phrase     = RELATION_PHRASES[ablation_relation]
    inv_rel    = INVERSE_REL[ablation_relation]
    inv_phrase = RELATION_PHRASES[inv_rel]

    # compile one regex that matches:
    #   Is the <SUBJ> {phrase} the <OBJ>?
    pattern = re.compile(
        rf"^Is the (.+?) {re.escape(phrase)} the (.+?)\?$"
    )

    changed = 0
    for q in questions:
        if q.get('relation_type') != ablation_relation:
            continue

        m = pattern.match(q['question'])
        if not m:
            # skip any unexpected format
            continue

        subj, obj = m.group(1), m.group(2)
        # rebuild question and flip relation_type
        q['question']      = f'Is the {obj} {inv_phrase} the {subj}?'
        q['relation_type'] = inv_rel
        # answer stays the same
        changed += 1

    print(f'Inverted {changed} “{ablation_relation}” → “{inv_rel}” out of {len(questions)} questions.')
    write_json(out_path, questions)
    print(f'Ablated dataset saved to {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Invert all LEFT (or any) questions into their inverse.'
    )
    parser.add_argument('--split', choices=['train','val','test'], default='train')
    parser.add_argument('--relation', choices=RELATION_TYPES, default='left')
    args = parser.parse_args()
    ablate_dataset(args.split, args.relation)
