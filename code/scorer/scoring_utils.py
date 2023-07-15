from collections import Counter


def list_to_span(l):
  assert(len(l) == 2)
  return (l[0], l[1])


def update(link_set, counter):
  link_set = Counter(link_set)
  for link, count in link_set.items():
    role = link[2]
    counter[role] += count


def compute_metrics(c, m, o):
  p = float(c) / float(c + m) if (c + m) else 0.0
  r = float(c) / float(c + o) if (c + o) else 0.0
  f1 = 2.0 * p * r  / (p + r) if (p + r) else 0.0 
  return (100.0 * p, 100.0 * r, 100.0 * f1)


def update_sentence_breakdowns(intersection, missing, overpred,
                               sentence_breakdowns, span_to_sent):
  def split_to_sent(span_list):
    sent_idx = [span_to_sent(span[1]) - span_to_sent(span[0]) + 2
                for span in span_list]
    return [Counter(
      {span: count
       for sent_id, (span, count) in zip(sent_idx, span_list.items())
       if sent_id == i}
    ) for i in range(5)]

  i_split = split_to_sent(intersection)
  m_split = split_to_sent(missing)
  o_split = split_to_sent(overpred)
  
  for i in range(5):
    update(i_split[i], sentence_breakdowns[i]["correct"])
    update(m_split[i], sentence_breakdowns[i]["missing"])
    update(o_split[i], sentence_breakdowns[i]["overpred"])


def compute_from_counters(correct, missing, overpred):
  total_correct = sum(correct.values())
  total_missing = sum(missing.values())
  total_overpred = sum(overpred.values())
  total_gold = total_correct + total_missing
  total_pred = total_correct + total_overpred
  precision, recall, f1 = compute_metrics(total_correct,
                                          total_missing,
                                          total_overpred)
  return precision, recall, f1, (total_gold, total_pred)


def compute_confusion(counter, intersection, missing, overpred):
  # Intersections are all correct, along diagonal
  for (e, a, r) in intersection:
    counter[r][r] += 1
  for (e1, a1, r_miss) in missing:
    for (e2, a2, r_over) in overpred:
      if e1 == e2 and a1 == a2:
        # This aggressively double counts errors
        counter[r_miss][r_over] += 1

        
def print_confusion(confusion):
  """This prints a normalized confusion matrix (from 0 to 100) """
  gold_counts = [(key, sum(row.values())) for key, row in confusion.items()]
  confusion_keys = [key for key, _ in sorted(gold_counts, key=lambda x: -x[1])]
  print (",".join([""] + confusion_keys))
  confusion_matrix = [",".join([r1] + ["{}".format(int(100 * confusion[r1][r2]/sum(confusion[r1].values())))
                                       for r2 in confusion_keys])
                      for r1 in confusion_keys]
  print ("\n".join(confusion_matrix))


def print_table(role_table, totals):
  """This prints a table with roles ordered by decreased role 
  frequency (according to the data)
  """
  table = []
  column_names = ["ROLE", "CORRECT", "MISSING", "OVERPRED", "F1"]
  table.append(column_names)
  sorted_roles = sorted(
    role_table.keys(),
    key=lambda r: (role_table[r]["CORRECT"] + role_table[r]["MISSING"]),
    reverse=True)
  for role in sorted_roles:
    table.append([role,
                  str(int(role_table[role]["CORRECT"])),
                  str(int(role_table[role]["MISSING"])),
                  str(int(role_table[role]["OVERPRED"])),
                  "{:.2f}".format(role_table[role]["F1"])])
  table.append(["TOTAL:",
                str(int(totals["CORRECT"])),
                str(int(totals["MISSING"])),
                str(int(totals["OVERPRED"])),
                "{:.2f}".format(totals["F1"])])
  for row in table:
    print("{:20.20}    {:8.8}    {:8.8}    {:8.8}    {}".format(*row))
