B = 79
S = 171
db = [[0 for _ in range(S)] for _ in range(B)]

with open('P_0_B_1_33_S_1_171.log') as f:
  for line in f.readlines():
    d = {}
    for kv in line.split():
      kv = kv.strip()
      k, v = kv.split('=')
      d[k] = v
    db[int(d['B'])][int(d['S'])] = float(d['BS_thr'])

print(','.join([str(s) for s in range(S)]))
for b in range(1, B):
  print(','.join([str(b)] + [str(db[b][s]) for s in range(1, S)]))