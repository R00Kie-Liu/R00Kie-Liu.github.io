#!/usr/bin/env python3
"""Pin and self-host the existing icon, highlighting and MathJax resources."""
from io import BytesIO
from pathlib import Path
import re
import tarfile
from urllib.parse import urljoin
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/site-libraries'
OUTPUT = ROOT / 'assets/lib'


def download(url):
    import hashlib
    path = CACHE / hashlib.sha256(url.encode()).hexdigest()
    if not path.exists():
        with urlopen(url, timeout=40) as response:
            path.write_bytes(response.read())
    return path.read_bytes()


def icon_css(url, folder):
    directory = OUTPUT / folder
    css = download(url).decode()
    for source in set(re.findall(r'url\(([^)]+)\)', css)):
        source = source.strip('"\'')
        target = source.split('?')[0].split('#')[0]
        if target.startswith('data:'):
            continue
        destination = (directory / 'css' / target).resolve()
        if not destination.is_relative_to(directory.resolve()):
            raise ValueError('Unexpected vendor font path')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(download(urljoin(url, source)))
    (directory / 'css').mkdir(parents=True, exist_ok=True)
    (directory / 'css' / url.rsplit('/', 1)[-1]).write_text(css)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    icon_css('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css', 'fontawesome-6.4.2')
    icon_css('https://cdnjs.cloudflare.com/ajax/libs/academicons/1.8.6/css/academicons.min.css', 'academicons-1.8.6')
    for name, url in {
        'fontawesome-6.4.2/LICENSE.txt': 'https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.4.2/LICENSE.txt',
        'academicons-1.8.6/README.md': 'https://raw.githubusercontent.com/jpswalsh/academicons/master/README.md',
        'highlight-11.9.0/highlight.min.js': 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js',
        'highlight-11.9.0/github.min.css': 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css',
        'highlight-11.9.0/github-dark.min.css': 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css',
        'highlight-11.9.0/LICENSE': 'https://raw.githubusercontent.com/highlightjs/highlight.js/11.9.0/LICENSE',
    }.items():
        path = OUTPUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(download(url))
    archive = download('https://registry.npmjs.org/mathjax/-/mathjax-3.2.2.tgz')
    with tarfile.open(fileobj=BytesIO(archive), mode='r:gz') as package:
        for member in package.getmembers():
            # Keep the selected renderer plus its autoload, accessibility and UI
            # components. Other renderer bundles are not used by this site.
            relative = member.name.removeprefix('package/')
            if not member.isfile() or not (relative == 'LICENSE' or relative.startswith('es5/')):
                continue
            if relative.startswith('es5/') and ('/output/svg/' in relative or relative.endswith('.map')):
                continue
            if relative.startswith('es5/') and relative.count('/') == 1 and relative != 'es5/tex-chtml.js':
                continue
            path = OUTPUT / 'mathjax-3.2.2' / relative
            if not path.resolve().is_relative_to((OUTPUT / 'mathjax-3.2.2').resolve()):
                raise ValueError('Unexpected archive path')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(package.extractfile(member).read())
    print('Pinned icons, highlight.js 11.9.0 and MathJax 3.2.2 are ready.')


if __name__ == '__main__':
    main()
