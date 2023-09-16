import numpy as np
with open('../llama_mp/hellaswag_full.log') as f:
  lines = f.readlines()
  lines = lines[14 : : 6]
  assert len(lines) == 10042
  count = 0
  for line in lines:
    _, normed_results, gold = line.strip().split(']')
    _, normed_results = normed_results.split('[')
    normed_results = [float(s) for s in normed_results.split()]
    gold = np.argmax(normed_results)
    print(normed_results, ';', gold)
    if np.argmax(normed_results) == gold:
      count += 1

#print(count)