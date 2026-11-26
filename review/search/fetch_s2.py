"""Systematic search, Semantic Scholar bulk search API (title+abstract), run 2026-11-25."""
import urllib.parse, urllib.request, json, csv, time
q = '("multi-task" | multitask | "panoptic driving perception") + ("drivable area" | "drivable region") + lane'
url = 'https://api.semanticscholar.org/graph/v1/paper/search/bulk?' + urllib.parse.urlencode(
    {'query': q, 'year': '2022-2026', 'fields': 'title,year,venue,externalIds,abstract,citationCount,publicationTypes'})
for attempt in range(5):
    try:
        d = json.load(urllib.request.urlopen(url)); break
    except Exception as e:
        print('retry', e); time.sleep(5)
rows = d.get('data', [])
print('S2 total', d.get('total'), len(rows))
with open('s2_records.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['source','id','doi','year','type','venue','cited_by','title','abstract'])
    for r in rows:
        ext = r.get('externalIds') or {}
        w.writerow(['s2', r['paperId'], ext.get('DOI',''), r.get('year'), ';'.join(r.get('publicationTypes') or []),
                    r.get('venue',''), r.get('citationCount',0), r['title'], r.get('abstract') or ''])
