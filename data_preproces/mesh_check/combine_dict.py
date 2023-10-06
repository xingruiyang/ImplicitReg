import json
import os

# split_filenames = [
#     'sv2_chairs_train.json',
#     'sv2_lamps_train.json',
#     'sv2_planes_train.json',
#     'sv2_sofas_train.json',
#     'sv2_tables_train.json'
# ]

split_filenames = [
    'sv2_chairs_test.json',
    'sv2_lamps_test.json',
    'sv2_planes_test.json',
    'sv2_sofas_test.json',
    'sv2_tables_test.json'
]

dicts = dict()
for split in split_filenames:
    dicts.update(json.load(open(split, 'r')))

json.dump(dicts, open('out.json', 'w'))
