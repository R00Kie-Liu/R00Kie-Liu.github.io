This is the latest version of my homepage's source code. Feel free to use and share.
<br />
For more details, please refer to this repository: <https://github.com/yaoyao-liu/minimal-light>.

### Using Locally with Jekyll

You need to install [Ruby](https://www.ruby-lang.org/en/) and [Jekyll](https://jekyllrb.com/) fisrt.

Install and run:

```bash
bundle install
bundle exec jekyll server
```
View the live page using `localhost`:
<http://localhost:4000>. You can get the html files in the `_site` folder.

### Blog link previews

External links in articles show a preview before opening a new tab. Internal
links navigate directly. Titles and descriptions come from public webpage
metadata or arXiv abstracts; readers' browsers never scrape destination sites.
After a post is published, the `Refresh link previews` workflow renders the posts,
refreshes stale metadata, commits the cache, and requests a GitHub Pages rebuild.
It can also be run manually from the Actions tab. Successful entries are reused
for 30 days; unavailable pages are retried after one day on the next run. Failed
refreshes preserve previously fetched metadata.

To refresh locally after building Jekyll:

```bash
python3 scripts/update-link-previews.py
python3 -m unittest discover -s tests -p 'test_link_previews.py'
```

The refresh history is `_data/link-previews.json`; the public metadata is
split into 16 small JSON files under `assets/link-previews/`. A file is requested
only when a reader clicks an external link and is then reused on that page.
Keep author overrides with `data-preview-title` and `data-preview-description` on
an HTML link; `data-no-preview` opts an individual link out of previews.

### Static assets and loading performance

Reading fonts, icon fonts, highlight.js 11.9.0, and MathJax 3.2.2 are served by
this site. Font outlines and weights are unchanged. Small Noto Serif SC subsets
cover the current rendered text; the complete self-hosted font remains a fallback
for characters introduced by future posts. No network font service is needed.
The upstream font CSS snapshot is pinned in `scripts/font-source.css`, and font
licenses ship beside the files.

Publication and technical-report thumbnails use responsive WebP images. Article
images retain their original resolution and zoom behavior. Large animations show
a real first-frame poster until the reader scrolls within 300px; clicking to zoom
always loads the original animation. A no-JavaScript fallback shows the original.
The small poster is a loading optimization, not a replacement for the animation.

All generated assets are committed, so GitHub Pages needs no extra build plugin.
New posts work immediately: new characters use the complete font fallback, and
unprocessed images still get native lazy loading. To refresh subsets and media
after a larger content update, first build Jekyll, then run:

```bash
python3 -m pip install -r scripts/requirements-assets.txt
python3 scripts/vendor-fonts.py
python3 scripts/optimize-site-images.py
# Rebuild Jekyll to include the generated assets and data.
```

`scripts/vendor-libraries.py` restores the pinned third-party distributions.
Analytics, the visitor globe, and comments remain optional external services.

For repeatable before/after measurements, build both versions, serve them with
the same local gzip server, install Playwright, and use a Chromium executable:

```bash
python3 scripts/serve-static.py --directory /tmp/site-performance-before --port 4173
python3 scripts/serve-static.py --directory _site --port 4174
# Run the servers in separate terminals, then measure cold loads:
python3 scripts/measure-site-performance.py \
  --before http://127.0.0.1:4173 --after http://127.0.0.1:4174 \
  --repeat 3 --chromium /usr/bin/chromium
```

Use `--width 390` for mobile, or `--kbps 4096 --latency 100 --cpu 4` for a
simulated slower connection/device. Results are local comparative measurements,
not promises about deployed visitors' networks. Optional analytics/comments are
excluded equally from both versions; raw resource costs are saved in the report.

### Google Scholar Crawler

The instructions for the Google Scholar crawler can be found in [this repository](https://github.com/RayeRen/acad-homepage.github.io).
<br>
Before using that, you need to change the Google Scholar ID in the following file:
https://github.com/yaoyao-liu/yaoyao-liu.github.io/blob/7d16d828a229580815428782fb74d937710eb50e/google_scholar_crawler/main.py#L7


### Acknowledgements

This project uses the source code from the following repositories:

* [pages-themes/minimal](https://github.com/pages-themes/minimal)

* [orderedlist/minimal](https://github.com/yaoyao-liu/minimal-light)

