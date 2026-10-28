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

### Google Scholar Crawler

The instructions for the Google Scholar crawler can be found in [this repository](https://github.com/RayeRen/acad-homepage.github.io).
<br>
Before using that, you need to change the Google Scholar ID in the following file:
https://github.com/yaoyao-liu/yaoyao-liu.github.io/blob/7d16d828a229580815428782fb74d937710eb50e/google_scholar_crawler/main.py#L7


### Acknowledgements

This project uses the source code from the following repositories:

* [pages-themes/minimal](https://github.com/pages-themes/minimal)

* [orderedlist/minimal](https://github.com/yaoyao-liu/minimal-light)

