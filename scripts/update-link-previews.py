#!/usr/bin/env python3
"""Cache public link metadata from rendered posts; no third-party dependencies."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from html import unescape
from html.parser import HTMLParser
import ipaddress
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import urljoin, urlsplit, urlunsplit, quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 1024 * 1024
USER_AGENT = 'HuabinLiu-LinkPreview/1.0 (+https://r00kie-liu.github.io/)'
ATOM = '{http://www.w3.org/2005/Atom}'


def clean(value, limit):
    value = re.sub(r'\s+', ' ', unescape(value or '')).strip()
    return value if len(value) <= limit else value[:limit - 1].rstrip() + '…'


def normalize_url(value, base=''):
    try:
        url = urlsplit(urljoin(base, value))
        host = (url.hostname or '').lower().encode('idna').decode('ascii')
        if url.scheme not in ('http', 'https') or not host or url.username or url.password:
            return None
        if host == 'localhost' or host.endswith(('.localhost', '.local', '.internal')):
            return None
        try:
            if not ipaddress.ip_address(host).is_global:
                return None
        except ValueError:
            pass
        port = url.port
        netloc = '[' + host + ']' if ':' in host else host
        if port and (url.scheme, port) not in (('http', 80), ('https', 443)):
            netloc += ':' + str(port)
        return urlunsplit((url.scheme, netloc, url.path or '/', url.query, ''))
    except (ValueError, UnicodeError):
        return None


class PostLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.anchor = None
        self.links = []
        self.canonical = ''

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == 'link' and attrs.get('rel') == 'canonical':
            self.canonical = attrs.get('href', '')
        if tag == 'div':
            if self.depth:
                self.depth += 1
            elif 'post-content' in attrs.get('class', '').split():
                self.depth = 1
        if not self.depth:
            return
        if tag == 'a':
            href = attrs.get('href', '')
            self.anchor = href if href and not href.startswith('#') and not any(
                key in attrs for key in ('download', 'data-no-preview')) else None
        if tag == 'img':
            self.anchor = None

    def handle_endtag(self, tag):
        if tag == 'a' and self.anchor:
            self.links.append(self.anchor)
            self.anchor = None
        if tag == 'div' and self.depth:
            self.depth -= 1


class Metadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta = {}
        self.in_title = False
        self.title = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == 'meta':
            key = (attrs.get('property') or attrs.get('name') or '').lower()
            if attrs.get('content'):
                self.meta.setdefault(key, attrs['content'])
        if tag == 'title':
            self.in_title = True

    def handle_endtag(self, tag):
        if tag == 'title':
            self.in_title = False

    def handle_data(self, value):
        if self.in_title:
            self.title.append(value)

    def result(self):
        title = clean(next((self.meta[k] for k in ('og:title', 'twitter:title', 'citation_title')
                            if self.meta.get(k)), ''.join(self.title)), 240)
        description = clean(next((self.meta[k] for k in ('og:description', 'description', 'twitter:description')
                                  if self.meta.get(k)), ''), 600)
        if re.match(r'^(just a moment|access denied|attention required|sign in|404|page not found)\b', title, re.I):
            return {}
        return {'title': title, 'description': description} if title else {}


def collect_urls(site_dir, site_url):
    origin = urlsplit(site_url).netloc.lower()
    urls = set()
    for path in sorted(site_dir.rglob('*.html')):
        parser = PostLinks()
        parser.feed(path.read_text(encoding='utf-8'))
        base = urljoin(site_url, parser.canonical or path.relative_to(site_dir).as_posix())
        for href in parser.links:
            url = normalize_url(href, base)
            if url and urlsplit(url).netloc.lower() != origin:
                urls.add(url)
    return urls


def arxiv_id(url):
    parsed = urlsplit(url)
    if parsed.hostname not in ('arxiv.org', 'www.arxiv.org', 'export.arxiv.org'):
        return None
    match = re.match(r'^/(?:abs|pdf|html)/(.+?)(?:\.pdf)?/?$', parsed.path)
    value = match.group(1) if match else ''
    return value if re.fullmatch(r'(?:\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})(?:v\d+)?', value) else None


def read_response(url):
    request = Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'text/html,application/atom+xml;q=0.9'})
    with urlopen(request, timeout=10) as response:
        if not normalize_url(response.geturl()):
            raise ValueError('Non-public redirect')
        data = response.read(MAX_BYTES)
        charset = response.headers.get_content_charset() or 'utf-8'
        return data.decode(charset, errors='replace'), response.headers.get_content_type()


def parse_arxiv(xml, requested):
    entries = {}
    feed = ET.fromstring(xml)
    for entry in feed.findall(ATOM + 'entry'):
        identifier = arxiv_id(entry.findtext(ATOM + 'id', ''))
        if not identifier:
            continue
        title = clean(entry.findtext(ATOM + 'title'), 240)
        if title:
            value = {'title': title, 'description': clean(entry.findtext(ATOM + 'summary'), 600)}
            entries[identifier] = value
            base = re.sub(r'v\d+$', '', identifier)
            # A request without a version asks arXiv for the latest revision.
            if base in requested:
                entries[base] = value
    return {key: entries[key] for key in requested if key in entries}


def refresh(urls, cache, now, workers=6):
    due = [url for url in sorted(urls) if now - cache.get(url, {}).get('checked_at', 0) >= 86400
           and (not cache.get(url, {}).get('fetched_at') or now - cache[url]['fetched_at'] >= 30 * 86400)]
    papers = {}
    other = []
    for url in due:
        identifier = arxiv_id(url)
        if identifier:
            papers.setdefault(identifier, []).append(url)
        else:
            other.append(url)
    success = 0

    def save(url, result):
        nonlocal success
        previous = cache.get(url, {})
        # Keep previously fetched metadata when a refresh fails.
        cache[url] = {**previous, 'checked_at': now}
        if result.get('title'):
            cache[url].update(result, fetched_at=now)
            success += 1

    ids = sorted(papers)
    for start in range(0, len(ids), 40):
        batch = ids[start:start + 40]
        try:
            xml, _ = read_response('https://export.arxiv.org/api/query?id_list=' + quote(','.join(batch), safe=',/') + '&max_results=40')
            values = parse_arxiv(xml, batch)
            for identifier in batch:
                for url in papers[identifier]:
                    save(url, values.get(identifier, {}))
        except Exception as error:
            print('arXiv batch unavailable (' + type(error).__name__ + '); trying abstract pages.', flush=True)
            other.extend(url for identifier in batch for url in papers[identifier])
        print('arXiv:', min(start + 40, len(ids)), '/', len(ids), flush=True)
        if start + 40 < len(ids):
            time.sleep(3)

    host_locks = {urlsplit(url).hostname: threading.Semaphore(2) for url in other}

    def fetch(url):
        with host_locks[urlsplit(url).hostname]:
            try:
                identifier = arxiv_id(url)
                target = 'https://arxiv.org/abs/' + identifier if identifier else url
                if re.search(r'\.(?:pdf|zip|gz|png|jpe?g|gif|webp|svg|mp4)$', urlsplit(target).path, re.I):
                    return url, {}
                text, content_type = read_response(target)
                if content_type not in ('text/html', 'application/xhtml+xml'):
                    return url, {}
                parser = Metadata()
                parser.feed(text)
                return url, parser.result()
            except Exception:
                return url, {}
            finally:
                time.sleep(0.3)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, future in enumerate(as_completed([pool.submit(fetch, url) for url in other]), 1):
            url, result = future.result()
            save(url, result)
            if count % 20 == 0 or count == len(other):
                print('Webpages:', count, '/', len(other), flush=True)
    print('Refreshed:', len(due), '| fetched:', success, '| unavailable:', len(due) - success, flush=True)
    return {url: cache[url] for url in sorted(urls) if url in cache}


def write_data(cache, output):
    output.mkdir(parents=True, exist_ok=True)
    shards = {key: {} for key in '0123456789abcdef'}
    for url, value in sorted(cache.items()):
        if value.get('title'):
            key = hashlib.sha256(url.encode()).hexdigest()[0]
            shards[key][url] = {k: value.get(k, '') for k in ('title', 'description')}
    for key, values in shards.items():
        (output / (key + '.json')).write_text(json.dumps(values, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-dir', type=Path, default=ROOT / '_site')
    parser.add_argument('--site-url', default='https://r00kie-liu.github.io/')
    parser.add_argument('--cache', type=Path, default=ROOT / '_data/link-previews.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'assets/link-previews')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not args.site_dir.is_dir():
        parser.error('Build Jekyll before refreshing link metadata.')
    urls = collect_urls(args.site_dir, args.site_url)
    if not urls:
        parser.error('No external article links found; refusing to clear the cache.')
    cache = json.loads(args.cache.read_text()) if args.cache.exists() else {}
    cache = refresh(urls, cache, int(time.time()), args.workers)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_data(cache, args.output)
    print('Cached metadata:', sum(bool(value.get('title')) for value in cache.values()), '/', len(urls))


if __name__ == '__main__':
    main()
