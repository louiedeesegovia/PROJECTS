import sys, json, glob, numpy as np
for pat in sys.argv[1:]:
    files = sorted(glob.glob(pat))
    rows = [json.load(open(f))['test'] for f in files]
    print(pat, '->', len(rows), 'runs')
    for k in ['mse', 'emd2', 'srcc', 'lcc']:
        v = np.array([r[k] for r in rows])
        print('  %-5s %.4f ± %.4f' % (k, v.mean(), v.std(ddof=1)))
