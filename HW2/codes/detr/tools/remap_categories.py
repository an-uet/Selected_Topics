#!/usr/bin/env python3
"""Check and remap category ids in a COCO-style annotation JSON.

Usage:
  python remap_categories.py /path/to/annotations.json [--out out.json]

If category ids are not consecutive starting at 0, the script will remap them
to 0..C-1 and write the result to --out (defaults to annotations.remapped.json).
"""
import json
import sys
from pathlib import Path


def remap(path: Path, out: Path):
    data = json.loads(path.read_text())
    cat_ids = sorted({c['id'] for c in data.get('categories', [])})
    if not cat_ids:
        print('no categories found in', path)
        return
    print('found category ids min..max:', min(cat_ids), '..', max(cat_ids))
    if cat_ids == list(range(0, len(cat_ids))):
        print('already 0..N-1, nothing to do')
        out.write_text(json.dumps(data))
        return

    # build mapping from old id -> new id (0..C-1)
    mapping = {old: new for new, old in enumerate(cat_ids)}
    print('remapping categories:', mapping)

    for ann in data.get('annotations', []):
        if 'category_id' in ann:
            ann['category_id'] = mapping[ann['category_id']]

    # update categories list ids
    for c in data.get('categories', []):
        c['id'] = mapping[c['id']]

    out.write_text(json.dumps(data))
    print('wrote remapped annotations to', out)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: remap_categories.py /path/to/annotations.json [--out out.json]')
        sys.exit(1)
    p = Path(sys.argv[1])
    out = None
    if '--out' in sys.argv:
        i = sys.argv.index('--out')
        out = Path(sys.argv[i + 1])
    if out is None:
        out = p.with_name(p.stem + '.remapped.json')
    remap(p, out)
