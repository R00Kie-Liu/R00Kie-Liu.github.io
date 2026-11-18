#!/usr/bin/env python3
"""Exercise unlisted note discovery against a real Jekyll build."""
import argparse
from html.parser import HTMLParser
from pathlib import Path

FIXTURES = [
    ("2000-01-01-visibility-public-a.md", "public-a", False, 1),
    ("2000-01-02-visibility-hidden-x.md", "hidden-x", True, 2),
    ("2000-01-03-visibility-public-b.md", "public-b", False, 3),
    ("2000-01-04-visibility-hidden-y.md", "hidden-y", True, 4),
    ("2001-01-01-visibility-public-c.md", "public-c", False, None),
    ("2001-01-02-visibility-hidden-z.md", "hidden-z", True, None),
    ("2001-01-03-visibility-public-d.md", "public-d", False, None),
]
HIDDEN_TAG = "visibility-hidden-only-tag"


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.html = path.read_text(encoding="utf-8")
        self.links = []
        self.robots = ""
        self.stack = []
        self.feed(self.html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set(attrs.get("class", "").split())
        if tag == "meta" and attrs.get("name") == "robots":
            self.robots = attrs.get("content", "")
        if tag == "a":
            ancestors = set().union(*(s[1] for s in self.stack)) if self.stack else set()
            self.links.append((attrs.get("href", ""), classes, ancestors))
        if tag not in {"meta", "link", "img", "input", "br", "hr", "source", "wbr", "area", "base", "embed", "param", "track"}:
            self.stack.append((tag, classes))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def cards(self):
        return {url for url, classes, _ in self.links if "post-card" in classes}

    def navigation(self):
        return {url for url, _, ancestors in self.links if "post-nav" in ancestors}

    def back_links(self):
        return {url for url, classes, _ in self.links if "back-link" in classes}


def url(slug):
    return "/visibility-check/" + slug + "/"


def prepare(source):
    posts = source / "_posts"
    posts.mkdir(parents=True, exist_ok=True)
    for filename, slug, unlisted, order in FIXTURES:
        path = posts / filename
        if path.exists():
            raise RuntimeError("Refusing to overwrite " + str(path))
        fields = [
            "---", "layout: post", "title: Visibility fixture " + slug,
            "permalink: " + url(slug),
            "tags: [" + (HIDDEN_TAG if unlisted else "visibility-public-tag") + "]",
        ]
        if unlisted:
            fields.append("visibility: unlisted")
        if order is not None:
            fields.extend(["series: visibility-fixture-series", "series_order: " + str(order)])
        fields.extend(["---", "", "## First section", "", "A rendered test note.", "",
                       "## Second section", "", "Content for table-of-contents rendering.", ""])
        path.write_text("\n".join(fields), encoding="utf-8")
    print("Prepared seven test notes in an isolated build checkout.")


def check(site):
    public = Page(site / "blog/index.html")
    workbench = Page(site / "workbench/index.html")
    hidden_urls = {url(slug) for _, slug, unlisted, _ in FIXTURES if unlisted}
    public_urls = {url(slug) for _, slug, unlisted, _ in FIXTURES if not unlisted}
    assert public_urls <= public.cards(), "A public note is missing"
    assert not hidden_urls & public.cards(), "Unlisted note appeared in public cards"
    assert hidden_urls <= workbench.cards(), "Unlisted note is missing from Workbench"
    assert not public_urls & workbench.cards(), "Public note appeared in Workbench"
    assert HIDDEN_TAG not in public.html, "Hidden-only tag leaked into public topics/search"
    assert "/workbench/" not in public.html, "Workbench leaked into public navigation"
    assert "noindex" in workbench.robots and "nofollow" in workbench.robots
    home = (site / "index.html").read_text(encoding="utf-8")
    assert "/workbench/" not in home, "Workbench leaked into homepage navigation"

    pages = {}
    for _, slug, unlisted, _ in FIXTURES:
        page = Page(site / ("visibility-check/" + slug + "/index.html"))
        pages[slug] = page
        assert page.back_links() == ({"/workbench/"} if unlisted else {"/blog/"})
        if unlisted:
            assert "noindex" in page.robots and "nofollow" in page.robots
            assert "Unlisted" in page.html
            assert "giscus.app/client.js" not in page.html
            assert page.navigation() <= hidden_urls, "Unlisted navigation crosses into public notes"
            assert url(slug) not in public.html, "Unlisted URL leaked into public output"
            assert "Visibility fixture " + slug not in public.html
        else:
            assert "noindex" not in page.robots
            assert not page.navigation() & hidden_urls, "Public navigation links to an unlisted note"
            assert "giscus.app/client.js" in page.html

    assert pages["public-a"].navigation() == {url("public-b")}
    assert pages["public-b"].navigation() == {url("public-a")}
    assert pages["hidden-x"].navigation() == {url("hidden-y")}
    assert pages["hidden-y"].navigation() == {url("hidden-x")}
    assert url("public-d") in pages["public-c"].navigation()
    assert url("public-c") in pages["public-d"].navigation()

    for path in site.rglob("*.xml"):
        if not any(part in {"sitemap", "feed", "rss"} for part in path.stem.split("-")):
            continue
        text = path.read_text(encoding="utf-8")
        assert "/workbench/" not in text, "Workbench leaked into " + str(path)
        assert not any(value in text for value in hidden_urls), "Unlisted URL leaked into " + str(path)

    print("PASS: public lists, search data, tags, serial/general navigation, robots, comments and XML discovery.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "check"])
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--site", type=Path, default=Path("_site"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.source)
    else:
        check(args.site)
