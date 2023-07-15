import argparse
import json
import re
from collections import Counter, defaultdict
from constraints import Constraints
import scoring_utils as util

class Scorer(object):
  def __init__(self, args):
    self.role_string_mapping = {}
    self.roles = set()
    self.gold = self.read_gold_file(args.gold_file)
    if args.reuse_gold_format:
      self.pred = self.read_gold_file(args.pred_file, confidence=False)
    else:
      self.pred = self.read_preds_file(args.pred_file)
    self.constraints = Constraints(args.ontology_file)
      
  def get_role_label(self, role):
    if role in self.role_string_mapping:
      return self.role_string_mapping[role]
    else:
      # Each role is of the form evt###arg##role, we only want role
      role_string = re.split(r'\d+', role)[-1]
      # assert (role_string == role[11:])
      
      self.role_string_mapping[role] = role_string
      self.roles.add(role_string)
      return role_string

  def read_gold_file(self, file_path, confidence=False):
    """
    Returns dict mapping doc_key -> (pred, arg, role)
    """
    def process_example(json_blob):
      doc_key = json_blob["doc_key"]
      gold_evt = json_blob["gold_evt_links"]
      sents = json_blob["sentences"]
      sent_map = []
      for i, sent in enumerate(sents):
          for _ in sent:
            sent_map.append(i)
      def span_to_sent(span):
        # assumes span does not cross boundaries
        sent_start = sent_map[span[0]]
        sent_end = sent_map[span[1]]
        assert (sent_start == sent_end)
        return sent_start

      # There should only be one predicate
      evt_triggers = json_blob["evt_triggers"]
      assert (len(evt_triggers) == 1)

      evt_trigger = evt_triggers[0]
      evt_trigger_span = util.list_to_span(evt_trigger[:2])
      evt_trigger_types = set([evt_trigger_type[0]
                               for evt_trigger_type in evt_trigger[2]]) 

      gold_evt_links = [(util.list_to_span(arg[0]),
                         util.list_to_span(arg[1]),
                         self.get_role_label(arg[2])) for arg in gold_evt]
      if confidence:
        gold_evt_links = [(a, b, c, 0) for a, b, c in gold_evt_links]
      assert (all([arg[0] == evt_trigger_span
                   for arg in gold_evt_links]))
      return (doc_key, gold_evt_links, evt_trigger_types, span_to_sent)
      
    
    jsonlines = open(file_path, 'r').readlines()
    lines = [process_example(json.loads(line)) for line in jsonlines]
    file_dict = {doc_key: (evt_links, evt_trigger_types, span_to_sent)
                      for doc_key, evt_links, evt_trigger_types, span_to_sent
                      in lines}
    return file_dict

  def read_preds_file(self, file_path):
    """
    Ideally have only a single file reader
    Returns dict mapping doc_key -> (pred, arg, role)
    """
    def process_example(json_blob):
      doc_key = json_blob["doc_key"]
      pred_evt = json_blob["predictions"]
      # There should only be one predicate
      if len(pred_evt) == 0:
        return (doc_key, [], None)
      assert(len(pred_evt) == 1)
      pred_evt = pred_evt[0]
      # convention that the 0th one is the predicate span
      evt_span = util.list_to_span(pred_evt[0])
      evt_args = pred_evt[1:]
      pred_args = [(evt_span,
                    util.list_to_span(args[:2]),
                    args[2],
                    args[3])
                   for args in evt_args]
      return doc_key, pred_args, None 

    jsonlines = open(file_path, 'r').readlines()
    lines = [process_example(json.loads(line)) for line in jsonlines]
    file_dict = {doc_key: (evt_links, evt_trigger_types)
                 for doc_key, evt_links, evt_trigger_types
                 in lines}
    return file_dict

  def create_role_table(self, correct, missing, overpred):
    role_table = {}
    for role in self.roles:
      c = float(correct[role])
      m = float(missing[role])
      o = float(overpred[role])
      p, r, f1 = util.compute_metrics(c, m, o)
      role_table[role] = {'CORRECT': c,
                          'MISSING': m,
                          'OVERPRED': o,
                          'PRECISION': p,
                          'RECALL': r,
                          'F1': f1}
    total_c = sum(correct.values())
    total_m = sum(missing.values())
    total_o = sum(overpred.values())
    total_p, total_r, total_f1 = util.compute_metrics(total_c,
                                                      total_m,
                                                      total_o)
    totals = {'CORRECT': total_c,
              'MISSING': total_m,
              'OVERPRED': total_o,
              'PRECISION': total_p,
              'RECALL': total_r,
              'F1': total_f1}
    return (role_table, totals)
  
  def evaluate(self, constrained_decoding=True):
    self.metrics = None
    self.distance_metrics = None
    self.role_table = None
    self.confusion = None
    # Also computes confusion counters
    global_confusion = defaultdict(Counter)
    sentence_breakdowns = [{
      "correct": Counter(),
      "missing": Counter(),
      "overpred": Counter()
    } for i in range(5)]
    total_lost = 0

    global_correct = Counter()
    global_missing = Counter()
    global_overpred = Counter()
    for doc_key, (gold_structure, evt_type, span_to_sent) in self.gold.items():
      pred_structure = self.pred.get(doc_key, ([], None))[0]
      pred_structure, lost = self.constraints.filter_preds(
        pred_structure,
        evt_type,
        constrained_decoding)

      total_lost += lost
      pred_set = Counter(pred_structure)
      gold_set = Counter(gold_structure)
      assert(sum(pred_set.values()) == len(pred_structure))
      assert(sum(gold_set.values()) == len(gold_structure))
      intersection = gold_set & pred_set
      missing = gold_set - pred_set
      overpred = pred_set - gold_set
      # Update confusion and counters
      util.compute_confusion(global_confusion, intersection,
                              missing, overpred)
      util.update(intersection, global_correct)
      util.update(missing, global_missing)
      util.update(overpred, global_overpred)
      util.update_sentence_breakdowns(intersection, missing, overpred,
                                       sentence_breakdowns, span_to_sent)
    precision, recall, f1, _ = util.compute_from_counters(global_correct, 
                                                          global_missing,
                                                          global_overpred)
    distance_metrics = []
    for i in range(5):
      i_p, i_r, i_f1, counts = util.compute_from_counters(
        sentence_breakdowns[i]["correct"],
        sentence_breakdowns[i]["missing"],
        sentence_breakdowns[i]["overpred"]
      )
      distance_metrics.append((i, (i_p, i_r, i_f1), counts))
    self.metrics = {'precision': precision,
                    'recall': recall,
                    'f1': f1}
    self.distance_metrics  = distance_metrics
    self.role_table = self.create_role_table(global_correct,
                                             global_missing,
                                             global_overpred)
    return {"role_table": self.role_table,
            "confusion": global_confusion,
            "metrics": self.metrics,
            "distance_metrics": self.distance_metrics}
  
def run_evaluation(args):
  """This is a separate wrapper around args so that other programs
  can call evaluation without resorting to an os-level call
  """
  scorer = Scorer(args)
  return_dict = scorer.evaluate(constrained_decoding=args.cd)
  if args.confusion or args.do_all:
    util.print_confusion(return_dict['confusion'])
  if args.role_table or args.do_all:
    util.print_table(*return_dict['role_table'])
  if args.distance or args.do_all:
    for (i, (p, r, f1), (gold, pred)) in return_dict['distance_metrics']:
      print (" {} & {} & {:.1f} & {:.1f} & {:.1f} \\\\ [p r f1 {} gold/{} pred. ]".format(
        i - 2, pred, p, r, f1, gold, pred))
  if args.metrics or args.do_all:
    print ("Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(
      return_dict['metrics']['precision'],
      return_dict['metrics']['recall'],
      return_dict['metrics']['f1']))
  return return_dict


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-g', '--gold_file', type=str,
                      help='Gold file path')
  parser.add_argument('-p', '--pred_file', type=str, default=None,
                      help='Predictions file path')
  parser.add_argument('--reuse_gold_format', dest='reuse_gold_format',
                      default=False, action='store_true',
                      help="Reuse gold file format for pred file.")
  parser.add_argument('-t', '--ontology_file', type=str, default=None,
                      help='Path to ontology file')
  parser.add_argument('-cd', '--type_constrained_decoding', dest="cd",
                      default=False, action='store_true',
                      help="Use type constrained decoding" +
                      '(only possible when ontology file is given')
  parser.add_argument('--do_all', dest='do_all', default=False,
                      action='store_true', help="Do everything.")
  parser.add_argument('--metrics', dest='metrics', default=False,
                      action='store_true',
                      help="Compute overall p, r, f1.")
  parser.add_argument('--distance', dest='distance', default=False,
                      action='store_true',
                      help="Compute p, r, f1 by distance.")
  parser.add_argument('--role_table', dest='role_table', default=False,
                      action='store_true',
                      help="Compute p, r, f1 per role.")
  parser.add_argument('--confusion', dest='confusion', default=False,
                      action='store_true',
                      help="Compute an error confusion matrix.")
  
  args = parser.parse_args()
  return_dict = run_evaluation(args)
