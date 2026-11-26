"""Fetch only the BDD100K *validation* images from the official archive.

The official file http://dl.yf.io/bdd100k/bdd100k_images_100k.zip (5.67 GB)
contains train/val/test. This script reads the zip central directory with
HTTP range requests and downloads only the entries under
100k/val/ (10,000 JPEGs, contiguous bytes ~3.97-4.53 GB of the archive), in parallel. The bytes are
identical to those obtained by downloading and unzipping the full archive
(CRC-32 of every file is verified against the central directory).

Usage:
    python replication/fetch_bdd_val.py --out data/bdd/images/val2017
"""
import argparse
import io
import struct
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = 'http://dl.yf.io/bdd100k/bdd100k_images_100k.zip'


def get_range(start, end):
    req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == 4:
                raise


class HttpFile(io.RawIOBase):
    """Minimal seekable file over HTTP ranges, enough for zipfile to read the central directory."""

    def __init__(self, size):
        self.size, self.pos = size, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0:
            return b''
        data = get_range(self.pos, min(self.pos + n, self.size) - 1)
        self.pos += len(data)
        return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--prefix', default='100k/val/')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--list-only', action='store_true')
    a = ap.parse_args()

    size = int(urllib.request.urlopen(urllib.request.Request(URL, method='HEAD')).headers['Content-Length'])
    zf = zipfile.ZipFile(HttpFile(size))
    infos = [i for i in zf.infolist() if i.filename.startswith(a.prefix) and not i.is_dir()]
    offs = sorted(i.header_offset for i in infos)
    print(f'{len(zf.infolist())} entries in archive, {len(infos)} under {a.prefix}; '
          f'byte span {offs[0]}..{offs[-1]} ({(offs[-1] - offs[0]) / 1e9:.2f} GB)')
    if a.list_only:
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def fetch(info):
        dst = out / Path(info.filename).name
        if dst.exists() and dst.stat().st_size == info.file_size:
            return 0
        blob = get_range(info.header_offset, info.header_offset + 30 + 1024 + len(info.filename) + info.compress_size)
        sig, *_ = struct.unpack('<I', blob[:4])
        assert sig == 0x04034b50, 'bad local header'
        n_len, x_len = struct.unpack('<HH', blob[26:30])
        start = 30 + n_len + x_len
        raw = blob[start:start + info.compress_size]
        data = raw if info.compress_type == zipfile.ZIP_STORED else zlib.decompress(raw, -15)
        assert zlib.crc32(data) & 0xffffffff == info.CRC, f'CRC mismatch {info.filename}'
        dst.write_bytes(data)
        return len(data)

    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, n in enumerate(ex.map(fetch, infos), 1):
            done += n
            if i % 500 == 0:
                print(f'{i}/{len(infos)} files, {done / 1e6:.0f} MB', flush=True)
    print('done', len(list(out.glob('*.jpg'))), 'files in', out)


if __name__ == '__main__':
    main()
