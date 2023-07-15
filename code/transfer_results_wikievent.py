import os
import spacy
from spacy.tokens import Doc
import argparse
import json
from tqdm import tqdm

'''
IN:
    - golden span format
    - predictions span format
OUT:
    - golden head format
    - golden coref format
    - golden head-identification format
    - golden coref-identification format
    - predictions head format
    - predictions coref format
    - predictions head-identification format
    - predictions coref-identification format
'''


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

parser = argparse.ArgumentParser()
parser.add_argument("-ig", "--infile_golden")
parser.add_argument("-ip", "--infile_prediction")
parser.add_argument("-o", "--outdir")
parser.add_argument("-s", "--split")
args = parser.parse_args()

def get_head(doc, span_b, span_e):
    cur_i = span_b
    while doc[cur_i].head.i >= span_b and doc[cur_i].head.i <=span_e:
        if doc[cur_i].head.i == cur_i:
            # self is the head 
            break 
        else:
            cur_i = doc[cur_i].head.i
        
    return cur_i

content = {}
coref_content = {}
golden_data = []
prediction_data = []

with open(args.infile_golden) as f:
    for line in f:
        golden_data.append(json.loads(line))
with open(args.infile_prediction) as f:
    for line in f:
        prediction_data.append(json.loads(line))
for d in tqdm(golden_data):
    sentences = d['sentences']
    sent = []
    for s in sentences:
        sent.extend(s)
    content[d['doc_key']] = sent
    coref_content[d['doc_key']] = dict()
    coref = d['coref']
    # use the first span to denote the mentions that belong to the same entity
    for coref_mentions in coref:
        valid_spans = set()
        for coref_mention_span in coref_mentions:
            if coref_mention_span != []:
                valid_spans.add(tuple(coref_mention_span))
        valid_spans = list(valid_spans)
        if len(valid_spans) > 1:
            for valid_span in valid_spans:
                coref_content[d['doc_key']][valid_span] = valid_spans[0]

def make_head(outfile_golden, outfile_prediction):
    golden_new_data = []
    prediction_new_data = []

    # process for golden data
    for d in tqdm(golden_data):
        r = {}
        r['doc_key'] = d['doc_key']
        r['gold_evt_links'] = []
        sentences = d['sentences']
        r['sentences'] = sentences
        r['evt_triggers'] = d['evt_triggers']
        gold_evt_links = d['gold_evt_links']
        
        sent = []
        for s in sentences:
            sent.extend(s)
        doc = nlp(' '.join(sent))

        for gold_evt_link in gold_evt_links:
            trigger_span = gold_evt_link[0]
            role_name = gold_evt_link[2]
            span_b, span_e = gold_evt_link[1]
            span_head = get_head(doc, span_b, span_e)
            r['gold_evt_links'].append([trigger_span, [span_head, span_head], role_name])
        golden_new_data.append(r)
            
    with open(outfile_golden, 'w') as f:
        for d in golden_new_data:
            f.write(json.dumps(d)+'\n')
            
    # process for prediction data
    for d in tqdm(prediction_data):
        r = {}
        r['doc_key'] = d['doc_key']
        r['predictions'] = [[]]
        sent = content[d['doc_key']]
        doc = nlp(' '.join(sent))

        prediction = d['predictions'][0]
        trigger_span = prediction[0]
        r['predictions'][0].append(trigger_span)
        
        for pre in prediction[1:]:
            span_b, span_e = pre[0], pre[1]
            span_head = get_head(doc, span_b, span_e)
            r['predictions'][0].append([span_head, span_head, pre[2], pre[3]])
            
        prediction_new_data.append(r)

    with open(outfile_prediction, 'w') as f:
        for d in prediction_new_data:
            f.write(json.dumps(d)+'\n')

def make_coref(outfile_golden, outfile_prediction):
    golden_new_data = []
    prediction_new_data = []

    # process for golden data
    for d in tqdm(golden_data):
        r = {}
        r['doc_key'] = d['doc_key']
        r['gold_evt_links'] = []
        sentences = d['sentences']
        r['sentences'] = sentences
        r['evt_triggers'] = d['evt_triggers']
        gold_evt_links = d['gold_evt_links']

        for gold_evt_link in gold_evt_links:
            trigger_span = gold_evt_link[0]
            role_name = gold_evt_link[2]
            span = tuple(gold_evt_link[1])
            if span in coref_content[d['doc_key']]:
                span = coref_content[d['doc_key']][span]
            r['gold_evt_links'].append([trigger_span, [span[0], span[1]], role_name])
        golden_new_data.append(r)
            
    with open(outfile_golden, 'w') as f:
        for d in golden_new_data:
            f.write(json.dumps(d)+'\n')
            
    # process for prediction data
    for d in tqdm(prediction_data):
        r = {}
        r['doc_key'] = d['doc_key']
        r['predictions'] = [[]]

        prediction = d['predictions'][0]
        trigger_span = prediction[0]
        r['predictions'][0].append(trigger_span)
        
        for pre in prediction[1:]:
            span = (pre[0], pre[1])
            if span in coref_content[d['doc_key']]:
                span = coref_content[d['doc_key']][span]
            r['predictions'][0].append([span[0], span[1], pre[2], pre[3]])
            
        prediction_new_data.append(r)

    with open(outfile_prediction, 'w') as f:
        for d in prediction_new_data:
            f.write(json.dumps(d)+'\n')


def make_head_identification(outfile_golden, outfile_prediction):
    golden_new_data = []
    prediction_new_data = []
    role_name = 'n/a'

    # process for golden data
    for d in tqdm(golden_data):
        r = {}
        r['doc_key'] = d['doc_key']
        r['gold_evt_links'] = []
        sentences = d['sentences']
        r['sentences'] = sentences
        r['evt_triggers'] = d['evt_triggers']
        gold_evt_links = d['gold_evt_links']
        
        sent = []
        for s in sentences:
            sent.extend(s)
        doc = nlp(' '.join(sent))

        for gold_evt_link in gold_evt_links:
            trigger_span = gold_evt_link[0]
            span_b, span_e = gold_evt_link[1]
            span_head = get_head(doc, span_b, span_e)
            r['gold_evt_links'].append([trigger_span, [span_head, span_head], role_name])
        golden_new_data.append(r)
            
    with open(outfile_golden, 'w') as f:
        for d in golden_new_data:
            f.write(json.dumps(d)+'\n')
            
    # process for prediction data
    for d in tqdm(prediction_data):
        r = {}
        r['doc_key'] = d['doc_key']
        r['predictions'] = [[]]
        sent = content[d['doc_key']]
        doc = nlp(' '.join(sent))

        prediction = d['predictions'][0]
        trigger_span = prediction[0]
        r['predictions'][0].append(trigger_span)
        
        for pre in prediction[1:]:
            span_b, span_e = pre[0], pre[1]
            span_head = get_head(doc, span_b, span_e)
            r['predictions'][0].append([span_head, span_head, role_name, pre[3]])
            
        prediction_new_data.append(r)

    with open(outfile_prediction, 'w') as f:
        for d in prediction_new_data:
            f.write(json.dumps(d)+'\n')

def make_coref_identification(outfile_golden, outfile_prediction):
    golden_new_data = []
    prediction_new_data = []
    role_name = 'n/a'

    # process for golden data
    for d in tqdm(golden_data):
        r = {}
        r['doc_key'] = d['doc_key']
        r['gold_evt_links'] = []
        sentences = d['sentences']
        r['sentences'] = sentences
        r['evt_triggers'] = d['evt_triggers']
        gold_evt_links = d['gold_evt_links']

        for gold_evt_link in gold_evt_links:
            trigger_span = gold_evt_link[0]
            span = tuple(gold_evt_link[1])
            if span in coref_content[d['doc_key']]:
                span = coref_content[d['doc_key']][span]
            r['gold_evt_links'].append([trigger_span, [span[0], span[1]], role_name])
        golden_new_data.append(r)
            
    with open(outfile_golden, 'w') as f:
        for d in golden_new_data:
            f.write(json.dumps(d)+'\n')
            
    # process for prediction data
    for d in tqdm(prediction_data):
        r = {}
        r['doc_key'] = d['doc_key']
        r['predictions'] = [[]]

        prediction = d['predictions'][0]
        trigger_span = prediction[0]
        r['predictions'][0].append(trigger_span)
        
        for pre in prediction[1:]:
            span = (pre[0], pre[1])
            if span in coref_content[d['doc_key']]:
                span = coref_content[d['doc_key']][span]
            r['predictions'][0].append([span[0], span[1], role_name, pre[3]])
            
        prediction_new_data.append(r)

    with open(outfile_prediction, 'w') as f:
        for d in prediction_new_data:
            f.write(json.dumps(d)+'\n')

split = args.split
assert split in ['validation', 'test']

make_head(outfile_golden=os.path.join(args.outdir, '{}_golden_head.jsonlines'.format(split)), 
        outfile_prediction=os.path.join(args.outdir, '{}_predictions_head.jsonlines'.format(split)))
make_coref(outfile_golden=os.path.join(args.outdir, '{}_golden_coref.jsonlines'.format(split)), 
        outfile_prediction=os.path.join(args.outdir, '{}_predictions_coref.jsonlines'.format(split)))
make_head_identification(outfile_golden=os.path.join(args.outdir, '{}_golden_head_identification.jsonlines'.format(split)), 
        outfile_prediction=os.path.join(args.outdir, '{}_predictions_head_identification.jsonlines'.format(split)))
make_coref_identification(outfile_golden=os.path.join(args.outdir, '{}_golden_coref_identification.jsonlines'.format(split)), 
        outfile_prediction=os.path.join(args.outdir, '{}_predictions_coref_identification.jsonlines'.format(split)))
