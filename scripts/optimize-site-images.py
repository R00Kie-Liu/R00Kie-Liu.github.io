#!/usr/bin/env python3
"""Generate publication thumbnails, large-animation posters and image dimensions."""
import hashlib
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import unquote
from urllib.request import urlopen

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'assets/optimized'


class Images(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = set()

    def handle_starttag(self, tag, attributes):
        if tag != 'img':
            return
        src = dict(attributes).get('src', '')
        if src.startswith('https://arxiv.org/'):
            self.sources.add(src)
            return
        if src.startswith(('http:', 'https:', '//', 'data:')):
            return
        self.sources.add(unquote(src.removeprefix('../').removeprefix('./').lstrip('/')))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    images = Images()
    cache = ROOT / '_data/image_assets.json'
    if cache.exists():
        images.sources.update(json.loads(cache.read_text()))
    for page in (ROOT / '_site').rglob('*.html'):
        images.feed(page.read_text())
    result = {}
    for src in sorted(images.sources):
        if src.startswith('assets/optimized/'):
            continue
        if src.startswith('https://arxiv.org/'):
            cache_file = ROOT / '.cache/site-images' / hashlib.sha256(src.encode()).hexdigest()
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                if not cache_file.exists():
                    with urlopen(src, timeout=30) as response:
                        cache_file.write_bytes(response.read())
                with Image.open(cache_file) as image:
                    extension = {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp'}.get(image.format)
                    if not extension:
                        continue
                    name = hashlib.sha256(cache_file.read_bytes()).hexdigest()[:16] + '.' + extension
                    (OUTPUT / name).write_bytes(cache_file.read_bytes())
                    result[src] = {'width': image.width, 'height': image.height,
                                   'local': '/assets/optimized/' + name, 'source': src}
            except Exception as error:
                print('Keeping remote image:', src, type(error).__name__)
            continue
        path = ROOT / src
        if not path.is_file() or path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.gif', '.webp'):
            continue
        with Image.open(path) as image:
            data = {'width': image.width, 'height': image.height}
            if src.startswith('pub_img/'):
                original = ImageOps.exif_transpose(image).convert('RGB')
                for width in (200, 400, 800):
                    thumbnail = original.copy()
                    thumbnail.thumbnail((width, 1600), Image.Resampling.LANCZOS)
                    encoded = BytesIO()
                    thumbnail.save(encoded, 'WEBP', quality=88, method=6)
                    name = hashlib.sha256(encoded.getvalue()).hexdigest()[:16] + '-' + str(width) + '.webp'
                    (OUTPUT / name).write_bytes(encoded.getvalue())
                    data['src' + str(width)] = '/assets/optimized/' + name
                    data['w' + str(width)] = thumbnail.width
            elif image.format == 'GIF' and getattr(image, 'n_frames', 1) > 1 and path.stat().st_size > 256 * 1024:
                # A real first-frame poster preserves the picture while the
                # original animation waits until the reader scrolls near it.
                poster = image.convert('RGBA')
                poster.thumbnail((1000, 1600), Image.Resampling.LANCZOS)
                encoded = BytesIO()
                poster.save(encoded, 'WEBP', quality=90, method=6)
                name = hashlib.sha256(encoded.getvalue()).hexdigest()[:16] + '-poster.webp'
                (OUTPUT / name).write_bytes(encoded.getvalue())
                data['poster'] = '/assets/optimized/' + name
                data['full'] = '/' + src
            result[src] = data
    for name in ('favicon-sjtu', 'favicon-sjtu-dark'):
        with Image.open(ROOT / 'assets/img' / (name + '.png')) as image:
            image = image.convert('RGBA')
            image.thumbnail((48, 48), Image.Resampling.LANCZOS)
            image.save(OUTPUT / (name + '-48.png'), optimize=True)
    keep = {'favicon-sjtu-48.png', 'favicon-sjtu-dark-48.png'}
    for value in result.values():
        keep.update(Path(item).name for item in value.values()
                    if isinstance(item, str) and item.startswith('/assets/optimized/'))
    for path in OUTPUT.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()
    (ROOT / '_data/image_assets.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('Prepared', len(result), 'image entries; original full-size images are unchanged.')


if __name__ == '__main__':
    main()
