"""Systematic search, OpenAlex (title+abstract), run 2026-11-25."""
import json, urllib.parse, urllib.request, csv, sys, time
Q = '("multi-task" OR multitask OR "panoptic driving perception") AND ("drivable area" OR "drivable region" OR "drivable space") AND lane'
def get(cursor):
    flt = f'title_and_abstract.search:{Q},publication_year:2022-2026'
    url = 'https://api.openalex.org/works?' + urllib.parse.urlencode({'filter': flt, 'per-page': 200, 'cursor': cursor,
        'select': 'id,doi,title,publication_year,type,primary_location,abstract_inverted_index,cited_by_count'})
    return json.load(urllib.request.urlopen(url))
rows, cursor = [], '*'
while cursor:
    d = get(cursor); rows += d['results']; cursor = d['meta'].get('next_cursor')
    if not d['results']: break
print('OpenAlex count', d['meta']['count'], len(rows))
def abstract(inv):
    if not inv: return ''
    pos = {p: w for w, ps in inv.items() for p in ps}
    return ' '.join(pos[i] for i in sorted(pos))
with open('openalex_records.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['source','id','doi','year','type','venue','cited_by','title','abstract'])
    for r in rows:
        src = (r.get('primary_location') or {}).get('source') or {}
        w.writerow(['openalex', r['id'], r.get('doi') or '', r['publication_year'], r['type'], src.get('display_name',''),
                    r.get('cited_by_count',0), r['title'], abstract(r.get('abstract_inverted_index'))])
