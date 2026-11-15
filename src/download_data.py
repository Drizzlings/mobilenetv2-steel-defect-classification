"""Download the original NEU-CLS archive from the authors' Google Drive link."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://drive.google.com/uc?export=download&id=1NGlXT9sIaQpyxUoT6MLKm1Pr6x8oxOvc'

class FormParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.action = None; self.fields = {}
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'form': self.action = a.get('action')
        if tag == 'input' and a.get('name'): self.fields[a['name']] = a.get('value', '')

def main():
    folder = ROOT / 'data'; folder.mkdir(exist_ok=True)
    dest = folder / 'NEU-CLS.rar'
    if dest.exists():
        print(f'Already downloaded: {dest}'); return
    response = urllib.request.urlopen(URL, timeout=60)
    if 'text/html' in response.headers.get('Content-Type', ''):
        parser = FormParser(); parser.feed(response.read().decode('utf-8'))
        if not parser.action or urllib.parse.urlparse(parser.action).hostname != 'drive.usercontent.google.com':
            raise RuntimeError('Expected official Google Drive download confirmation form')
        response = urllib.request.urlopen(parser.action + '?' + urllib.parse.urlencode(parser.fields), timeout=120)
    if 'text/html' in response.headers.get('Content-Type', ''):
        raise RuntimeError('Download returned HTML instead of archive')
    temporary = dest.with_suffix('.part'); total = 0; digest = hashlib.sha256()
    with temporary.open('wb') as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk); digest.update(chunk); total += len(chunk)
            if total % (20 * 1024 * 1024) == 0: print(f'Downloaded {total // 1048576} MiB', flush=True)
    if temporary.read_bytes()[:4] != b'Rar!': raise RuntimeError('Unexpected archive format')
    temporary.replace(dest)
    provenance = {'source_page': 'https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/',
                  'download_url': URL, 'archive_sha256': digest.hexdigest(), 'archive_bytes': total,
                  'dataset': 'NEU-CLS original image classification archive'}
    (folder / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(provenance, indent=2))

if __name__ == '__main__': main()
