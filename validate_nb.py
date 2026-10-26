import json

p = 'e:/Bdtorrent/CHikni/work/notebooks/ml07_rule_baseline_ranked_queue.ipynb'
nb = json.load(open(p))
print('valid JSON notebook')
print('nbformat:', nb.get('nbformat'), '.', nb.get('nbformat_minor'))
print('cells:', len(nb['cells']))
for i, c in enumerate(nb['cells']):
    print('  cell %d: %s (%d lines)' % (i, c['cell_type'], len(c['source'])))
print('kernelspec:', nb['metadata']['kernelspec'])