#!/usr/bin/env python3
"""Serve a Jekyll build locally with gzip, for comparable cold-load measurements."""
import argparse
from functools import partial
from functools import lru_cache
import gzip
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path


@lru_cache(maxsize=512)
def compressed(path, modified):
    return gzip.compress(Path(path).read_bytes(), mtime=0)


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = Path(self.translate_path(self.path))
        if path.is_dir():
            if not self.path.split('?', 1)[0].endswith('/'):
                return super().send_head()
            if (path / 'index.html').is_file():
                path = path / 'index.html'
            else:
                return super().send_head()
        content_type = self.guess_type(str(path))
        if (path.is_file() and 'gzip' in self.headers.get('Accept-Encoding', '')
                and (content_type.startswith('text/') or content_type in ('application/javascript', 'application/json', 'image/svg+xml'))):
            data = compressed(str(path), path.stat().st_mtime_ns)
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Encoding', 'gzip')
            self.send_header('Vary', 'Accept-Encoding')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            return BytesIO(data)
        return super().send_head()

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--port', type=int, default=4173)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(args.directory)))
    print('Serving', args.directory, 'on', args.port, flush=True)
    server.serve_forever()
