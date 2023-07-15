import json
from tqdm import tqdm

def transfer(split, MAX_LEN = 600):
    data = []
    new_data = []
    coref = {}

    with open('{}.jsonl'.format(split)) as f:
        for line in f.readlines():
            data.append(json.loads(line))
    with open('coref/{}.jsonlines'.format(split)) as f:
        for line in f.readlines():
            d = json.loads(line)
            coref[d['doc_key']] = d['clusters']

    for d in tqdm(data):
        doc_id = d['doc_id']
        tokens = d['tokens']
        sentences = d['sentences']
        entity_mentions = d['entity_mentions']  # 一条数据中对应的实体mention
        event_mentions = d['event_mentions'] # 一条数据中所有事件的类型和相关触发词
        assert sum([len(s[:-1][0]) for s in sentences]) == len(tokens)
        
        entity_information = dict()
        for entity_mention in entity_mentions:
            entity_information[entity_mention['id']] = (entity_mention['start'], entity_mention['end']-1)  # mention对应在整篇文章中的span位置编号
        coref_entities = coref[doc_id]  # 相关的实体？
        
        for i, event_mention in enumerate(event_mentions):
            new_d = {}
            new_d['doc_key'] = '{}-{}'.format(doc_id, i)
            
            event_type = event_mention['event_type']
            trigger = event_mention['trigger']
            trigger_b, trigger_e, trigger_sent = trigger['start'], trigger['end']-1, trigger['sent_idx']

            # use trigger as center and expand leftward and rightward
            cur_len = len(sentences[trigger_sent][:-1][0])  # 这里的span是字符的长度
            context = [[tup[0] for tup in sentences[trigger_sent][:-1][0]]] # 触发词所在句子的具体上下文内容
            if cur_len > MAX_LEN:  # 如果句子长度超过最大长度
                left_bound = sum([len(s[:-1][0]) for s in sentences[:trigger_sent]])
                right_bound = sum([len(s[:-1][0]) for s in sentences[:trigger_sent+1]]) - 1
                left_w = trigger_b - 1
                right_w = trigger_e + 1
                cur_len = right_w - left_w - 1
                new_context = context[0][left_w-left_bound+1:right_w-left_bound]
                while cur_len < MAX_LEN:
                    if cur_len < MAX_LEN and left_w >= left_bound:
                        new_context = [context[0][left_w-left_bound]] + new_context
                        left_w -= 1
                        cur_len += 1
                    if cur_len < MAX_LEN and right_w <= right_bound:
                        new_context = new_context + [context[0][right_w-left_bound]]
                        right_w += 1
                        cur_len += 1
                offset = left_w + 1
                context = [new_context]
            else:
                left_sen = trigger_sent - 1
                right_sen = trigger_sent + 1
                prev_len = 0
                while cur_len < MAX_LEN and left_sen >= 0 and right_sen < len(sentences) and cur_len != prev_len:
                    prev_len = cur_len
                    if left_sen >= 0:
                        left_sen_tokens = [tup[0] for tup in sentences[left_sen][:-1][0]]
                        if cur_len + len(left_sen_tokens) <= MAX_LEN:
                            context = [left_sen_tokens] + context
                            left_sen -=1 
                            cur_len += len(left_sen_tokens)
                            
                    if right_sen < len(sentences):
                        right_sen_tokens = [tup[0] for tup in sentences[right_sen][:-1][0]]
                        if cur_len + len(right_sen_tokens) <= MAX_LEN:
                            context = context + [right_sen_tokens]
                            right_sen +=1 
                            cur_len += len(right_sen_tokens)
                offset = sum([len(s[:-1][0]) for s in sentences[:left_sen+1]])
            
            all_context = []  # 只有触发词的句子
            for c in context:
                all_context.extend(c)
            assert len(all_context) <= MAX_LEN

            new_d['evt_triggers'] = [[trigger_b-offset, trigger_e-offset, [[event_type, 1.0]]]]
            new_d['sentences'] = context
            new_d['gold_evt_links'] = []
            arguments = event_mention['arguments']
            for argument in arguments:
                entity_id = argument['entity_id']
                span_b, span_e = entity_information[entity_id]
                role = argument['role']
                # 注意减去offset之后可能会出现负数或者超过context长度！
                # 这个同样需要加入evaluation，并且模型不可能预测出来，否则对比不公平！
                new_d['gold_evt_links'].append([[trigger_b-offset, trigger_e-offset], [span_b-offset, span_e-offset], role]) 

                assert trigger_b-offset >=0 and trigger_b-offset < len(all_context) and trigger_e-offset >=0 and trigger_e-offset < len(all_context)
                assert tokens[trigger_b] == all_context[trigger_b-offset]
                assert tokens[trigger_e] == all_context[trigger_e-offset]

                if span_b-offset >= 0 and span_b-offset < len(all_context) and span_e-offset >= 0 and span_e-offset < len(all_context):
                    assert tokens[span_b:span_e+1] == all_context[span_b-offset:span_e-offset+1]
                else:
                    print('You cannot predict this one!')


            new_d['coref'] = []
            for coref_entity in coref_entities:
                r = []
                for entity in coref_entity:
                    entity_span_b, entity_span_e = entity_information[entity]
                    entity_span_b -= offset
                    entity_span_e -= offset
                    if entity_span_b >= 0 and entity_span_b < len(all_context) and entity_span_e >= 0 and entity_span_e < len(all_context):
                        # 只有这些是出现在这个取的context里面的 才是有用的 否则不可能取到这个span来做coref的！
                        r.append((entity_span_b, entity_span_e))
                    
                new_d['coref'].append(r)

            new_d['ent_spans'] = []
            for entity_mention in entity_mentions:
                entity_b, entity_e = entity_mention['start'], entity_mention['end']-1
                entity_b -= offset
                entity_e -= offset
                if entity_b >= 0 and entity_b < len(all_context) and entity_e >= 0 and entity_e < len(all_context):
                    new_d['ent_spans'].append([entity_b, entity_e, [["", 1.0]]])

            new_data.append(new_d)

    with open('transfer-{}.jsonl'.format(split), 'w') as f:
        for new_d in new_data:
            f.write(json.dumps(new_d)+'\n')

if __name__ == '__main__':
    for split in ['train', 'dev', 'test']:
        transfer(split)
