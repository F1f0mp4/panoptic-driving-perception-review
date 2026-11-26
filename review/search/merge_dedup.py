"""Merge the three record sets and remove duplicates (DOI or normalised title)."""
import csv, re
def norm(t): return re.sub(r'[^a-z0-9]', '', (t or '').lower())
recs = []
for fn in ['openalex_records.csv', 's2_records.csv', 'arxiv_records.csv']:
    recs += list(csv.DictReader(open(fn)))
print('identified total', len(recs))
seen = {}
for r in recs:
    k = norm(r['title']); d = (r['doi'] or '').lower().replace('https://doi.org/', '')
    hit = next((kk for kk, v in seen.items() if v['_t'] == k or (d and v['_d'] == d)), None)
    if hit: seen[hit]['sources'].add(r['source'])
    else:
        r['_t'], r['_d'], r['sources'] = k, d, {r['source']}
        seen[d if d and not d.startswith('10.48550') else k] = r
uniq = sorted(seen.values(), key=lambda r: (r['year'], r['title']))
print('after dedup', len(uniq))
with open('merged_dedup.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['rid','sources','doi','year','venue','cited_by','title','abstract'])
    for i, r in enumerate(uniq):
        w.writerow([f'R{i+1:03d}', '+'.join(sorted(r['sources'])), r['_d'], r['year'], r['venue'], r['cited_by'], r['title'], r['abstract']])
