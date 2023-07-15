import json
from tqdm import tqdm

# use TRAINING SET to make meta 

def make_meta(src_path, tgt_path):
    data = []
    eventtype2role = dict()

    with open(src_path) as f:
        for line in f:
            data.append(json.loads(line))
    for d in tqdm(data):
        event_mentions = d['event_mentions']
        for event_mention in event_mentions:
            event_type = event_mention['event_type']
            if event_type not in eventtype2role:
                eventtype2role[event_type] = set()
            arguments = event_mention['arguments']
            for argument in arguments:
                role = argument['role']
                eventtype2role[event_type].add(role)

    result = []
    for k, v in eventtype2role.items():
        r = [k, []]
        for vv in v:
            r[-1].append(vv)
        result.append(r)

    with open(tgt_path, 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    make_meta('wikievents/train.jsonl', 'meta.json')