import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('previews', Path(__file__).parents[1] / 'scripts/update-link-previews.py')
previews = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previews)


class LinkPreviewTests(unittest.TestCase):
    def test_metadata_priority_entities_and_plain_text(self):
        parser = previews.Metadata()
        parser.feed('''<title>Fallback</title><meta name="description" content="Fallback summary">
            <meta property="og:title" content="Paper &amp; code">
            <meta property="og:description" content="Research &lt;script&gt;alert(1)&lt;/script&gt;  summary">''')
        self.assertEqual(parser.result()['title'], 'Paper & code')
        self.assertEqual(parser.result()['description'], 'Research <script>alert(1)</script> summary')
        blocked = previews.Metadata()
        blocked.feed('<title>Just a moment...</title>')
        self.assertEqual(blocked.result(), {})

    def test_url_identity_matches_browser_and_skips_non_public_targets(self):
        self.assertEqual(previews.normalize_url('HTTPS://EXAMPLE.ORG:443#section'), 'https://example.org/')
        self.assertEqual(previews.normalize_url('../paper#abstract', 'https://example.org/blog/post'), 'https://example.org/paper')
        for url in ('javascript:alert(1)', 'mailto:a@example.org', 'https://localhost/',
                    'http://127.0.0.1/', 'https://user:secret@example.org/'):
            self.assertIsNone(previews.normalize_url(url), url)

    def test_discovery_uses_rendered_post_content_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'post.html').write_text('''<a href="https://nav.example/">Navigation</a>
                <link rel="canonical" href="https://site.example/2026/post.html">
                <div class="post-content"><div><a href="https://paper.example/#abstract">Paper</a></div>
                <a href="/other.html">Internal</a><a href="https://site.example/another.html">Absolute internal</a>
                <a href="#heading">Anchor</a><a href="https://image.example/full.png"><img src="thumb.png"></a>
                <a href="https://download.example/" download>Download</a>
                <a href="https://optout.example/" data-no-preview>Opt out</a></div>
                <a href="https://footer.example/">Footer</a>''')
            self.assertEqual(previews.collect_urls(path, 'https://site.example/'), {'https://paper.example/'})

    def test_arxiv_pdf_and_versioned_abstract(self):
        self.assertEqual(previews.arxiv_id('https://arxiv.org/pdf/2409.19606v1.pdf'), '2409.19606v1')
        xml = '''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
            <id>http://arxiv.org/abs/2409.19606v2</id><title>Hyper-Connections</title>
            <summary>Actual abstract\n with whitespace.</summary></entry></feed>'''
        values = previews.parse_arxiv(xml, ['2409.19606', '2409.19606v1', '2409.19606v2'])
        self.assertEqual(values['2409.19606']['description'], 'Actual abstract with whitespace.')
        self.assertNotIn('2409.19606v1', values)

    def test_cache_survives_network_failure_and_retries_later(self):
        url = 'https://example.org/'
        now = 40 * 86400
        old = {'title': 'Cached title', 'description': 'Cached description', 'fetched_at': 1, 'checked_at': 1}
        with patch.object(previews, 'read_response', side_effect=TimeoutError), patch.object(previews.time, 'sleep'):
            cache = previews.refresh({url}, {url: old.copy()}, now, workers=1)
            self.assertEqual(cache[url]['title'], old['title'])
            self.assertEqual(cache[url]['checked_at'], now)
            self.assertEqual(cache[url]['fetched_at'], 1)
            previews.refresh({url}, cache, now + 3600, workers=1)
            self.assertEqual(previews.read_response.call_count, 1)

    def test_fresh_cache_avoids_requests_and_writes_matching_shard(self):
        url = 'https://example.org/paper'
        now = 40 * 86400
        value = {'title': 'Paper title', 'description': 'Paper summary', 'fetched_at': now, 'checked_at': now}
        with patch.object(previews, 'read_response') as request:
            cache = previews.refresh({url}, {url: value}, now + 3600)
            request.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            previews.write_data(cache, output)
            self.assertEqual(len(list(output.glob('*.json'))), 16)
            shard = hashlib.sha256(url.encode()).hexdigest()[0]
            entry = json.loads((output / (shard + '.json')).read_text())[url]
            self.assertEqual(entry, {'title': 'Paper title', 'description': 'Paper summary'})
            self.assertNotIn('checked_at', entry)


if __name__ == '__main__':
    unittest.main()
