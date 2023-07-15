import argparse
from collections import Counter

class Constraints(object):
  def __init__(self, constraints_file=None):
    if constraints_file is not None:
      self.constraints = self._load_constraints(constraints_file)
    else:
      self.constraints = None

  def _load_constraints(self, constraints_file):
    lines = open(constraints_file, 'r').readlines()
    constraints = {}
    for line in lines:
      key = None
      role_labels = []
      counts = []
      for i, token in enumerate(line.split()):
        if i == 0:
          key = token
        elif i > 0 and i % 2 == 1:
          role_labels.append(token)
        else:
          counts.append(int(token))
      constraints[key] = Counter(dict(zip(role_labels, counts)))
    return constraints

  def filter_preds(self, structure, evt_type, use_type_constrained):
    if not use_type_constrained:
      return ([span[:3] for span in structure], 0)
    assert(len(evt_type) == 1)
    evt_type = list(evt_type)[0]
    if evt_type not in self.constraints:
      print ("not found: {}".format(evt_type))
      return ([span[:3] for span in structure], 0)
    evt_roles = self.constraints[evt_type]
    filtered_structures = []
    local_counter = Counter()
    for span in sorted(structure, key=lambda x: -x[3]):
      if local_counter[span[2]] < evt_roles[span[2]]:
        filtered_structures.append(span[:3])
        local_counter[span[2]] += 1
    return (filtered_structures, len(structure) - len(filtered_structures))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--ontology_file', type=str, default=None,
                      help='Path to ontology file')
  args = parser.parse_args()
  c = Constraints(args.ontology_file)
  print (c.constraints)
