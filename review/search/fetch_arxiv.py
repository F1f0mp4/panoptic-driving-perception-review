"""Systematic search, arXiv API (title+abstract), run 2026-11-25."""
import urllib.parse, urllib.request, csv, re, time
import xml.etree.ElementTree as ET
Q = '(abs:"multi-task" OR abs:multitask OR abs:"panoptic driving perception") AND (abs:"drivable area" OR abs:"drivable region") AND abs:lane'
url = 'http://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': Q, 'start': 0, 'max_results': 500})
x = urllib.request.urlopen(url).read()
ns = {'a': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
root = ET.fromstring(x)
rows = []
for e in root.findall('a:entry', ns):
    yr = int(e.find('a:published', ns).text[:4])
    doi = e.find('arxiv:doi', ns)
    jr = e.find('arxiv:journal_ref', ns)
    rows.append(['arxiv', e.find('a:id', ns).text, doi.text if doi is not None else '', yr, 'preprint',
                 jr.text if jr is not None else 'arXiv', '', re.sub(r'\s+', ' ', e.find('a:title', ns).text).strip(),
                 re.sub(r'\s+', ' ', e.find('a:summary', ns).text).strip()])
print('arXiv total', root.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults').text, 'in 2022-2026:', sum(r[3] >= 2022 for r in rows))
with open('arxiv_records.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['source','id','doi','year','type','venue','cited_by','title','abstract'])
    w.writerows([r for r in rows if 2022 <= r[3] <= 2026])
