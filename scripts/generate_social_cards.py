#!/usr/bin/env python3
"""Generate static blog sharing images and crawler-readable metadata."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
from html import escape
from io import BytesIO
from itertools import combinations
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urljoin, urlsplit

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1200, 630
START = "    <!-- Generated social preview: scripts/generate_social_cards.py -->"
END = "    <!-- End generated social preview -->"
BLOCK = re.compile(re.escape(START) + r".*?" + re.escape(END) + r"\n?", re.S)


@dataclass
class Post:
    path: Path
    source: str
    title: str
    author: str
    description: str
    image: Path | None
    image_alt: str
    url: str
    authored_description: bool
    authored_author: bool


def local_path(root, url):
    """Resolve site URLs without accepting remote assets or path traversal."""
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc:
        raise ValueError(f"Expected a local site URL: {url}")
    path = (root / unquote(parsed.path).lstrip("/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path is outside the site: {url}")
    return path


def text_of(element):
    return " ".join(element.stripped_strings) if element else ""


def read_posts(root):
    domain = (root / "CNAME").read_text().strip()
    if not re.fullmatch(r"[A-Za-z0-9.-]+", domain):
        raise ValueError("CNAME must contain a single domain")
    origin = f"https://{domain}"
    index = BeautifulSoup((root / "blog/index.html").read_text(), "html.parser")
    posts = []
    seen = set()
    # Commented-out entries and unlisted drafts are intentionally excluded.
    for link in index.select(".post-entry a.read-more[href]"):
        route = urlsplit(link["href"]).path
        if not re.fullmatch(r"/blog/[^/]+/", route):
            raise ValueError(f"Expected a blog post directory URL: {route}")
        path = local_path(root, route) / "index.html"
        if path in seen:
            continue
        seen.add(path)
        source = path.read_text()
        if source.count(START) != source.count(END) or source.count(START) > 1:
            raise ValueError(f"Malformed generated metadata block: {path}")
        clean = BLOCK.sub("", source)
        page = BeautifulSoup(clean, "html.parser")
        title = text_of(page.select_one("h1.post-header"))
        author_meta = page.select_one('meta[name="author"]')
        author = (author_meta.get("content", "") if author_meta else
                  text_of(page.select_one(".post-meta > span")))
        description_meta = page.select_one('meta[name="description"]')
        description = (description_meta.get("content", "") if description_meta else
                       text_of(page.select_one(".post-subtitle")))
        if not description:
            description = text_of(link.find_parent(class_="post-entry").find("p"))
        if not all((title, author, description)):
            raise ValueError(f"Missing title, author, or description in {path}")
        if not page.head or not re.search(r"</head\s*>", clean, flags=re.I):
            raise ValueError(f"Missing head in {path}")
        if page.select('meta[property^="og:"], meta[name^="twitter:"], link[rel="canonical"]'):
            raise ValueError(f"Move existing sharing/canonical tags into the generated block: {path}")
        hero = page.select_one(".hero-image img")
        image = None
        if hero:
            image_url = urljoin(route, hero["src"])
            image = local_path(root, image_url)
            if not image.is_file():
                raise ValueError(f"Missing hero image: {image}")
        posts.append(Post(path, clean, title, author, description, image,
                          hero.get("alt", "") if hero else "", origin + route,
                          description_meta is not None, author_meta is not None))
    if not posts:
        raise ValueError("No published blog entries found")
    return posts, domain


def font(root, filename, size):
    return ImageFont.truetype(str(root / "assets/fonts" / filename), size,
                              layout_engine=ImageFont.Layout.BASIC)


def wrap(text, face, max_width):
    words = text.split()
    # Balance line lengths so long titles don't leave a single word stranded.
    for count in range(1, min(3, len(words)) + 1):
        candidates = []
        for breaks in combinations(range(1, len(words)), count - 1):
            bounds = (0, *breaks, len(words))
            lines = [" ".join(words[a:b]) for a, b in zip(bounds, bounds[1:])]
            lengths = [face.getlength(line) for line in lines]
            if max(lengths) <= max_width:
                score = sum((max_width - length) ** 2 for length in lengths)
                candidates.append((score, lines))
        if candidates:
            return min(candidates, key=lambda candidate: candidate[0])[1]
    return None


def render_card(root, post, domain):
    if post.image:
        with Image.open(post.image) as original:
            canvas = ImageOps.fit(ImageOps.exif_transpose(original).convert("RGB"),
                                  (WIDTH, HEIGHT), Image.Resampling.LANCZOS).convert("RGBA")
        # Keep the painting visible above the title, darken the text region.
        shade = Image.new("RGBA", (WIDTH, HEIGHT))
        pixels = ImageDraw.Draw(shade)
        for y in range(HEIGHT):
            depth = y / (HEIGHT - 1)
            opacity = round(max(35 + 195 * depth ** 1.25,
                                120 * max(0, 1 - depth / 0.32)))
            pixels.line((0, y, WIDTH, y), fill=(9, 14, 16, opacity))
        canvas = Image.alpha_composite(canvas, shade)
        ink, muted = "#fff9ec", "#e3ded1"
    else:
        canvas = Image.new("RGBA", (WIDTH, HEIGHT), "#f0eee7")
        ink, muted = "#242520", "#57584f"

    draw = ImageDraw.Draw(canvas)
    draw.text((64, 46), "not so sparse", font=font(root, "Alegreya.ttf", 28),
              fill=muted, anchor="lt")
    # Adapt to long titles; never truncate the actual post title.
    for size in range(100, 41, -2):
        face = font(root, "AlegreyaSC-Regular.ttf", size)
        lines = wrap(post.title, face, WIDTH - 128)
        leading = round(size * 1.12)
        if lines and len(lines) <= 3 and len(lines) * leading <= 300:
            break
    else:
        raise ValueError(f"Title does not fit the social card: {post.title}")
    y = 477 - len(lines) * leading
    for line in lines:
        draw.text((64, y), line, font=face, fill=ink, anchor="lt")
        y += leading
    draw.line((64, 510, WIDTH - 64, 510), fill=muted, width=1)
    author_face = font(root, "Alegreya.ttf", 34)
    domain_face = font(root, "Alegreya.ttf", 25)
    if author_face.getlength(post.author) + domain_face.getlength(domain) > WIDTH - 180:
        raise ValueError(f"Author and domain do not fit: {post.author}")
    draw.text((64, 543), post.author, font=author_face, fill=ink, anchor="lt")
    draw.text((WIDTH - 64, 550), domain, font=domain_face, fill=muted, anchor="rt")
    output = BytesIO()
    canvas.convert("RGB").save(output, "JPEG", quality=92, subsampling=0, optimize=True)
    return output.getvalue()


def metadata(post, image_url):
    image_alt = f"{post.title} — {post.author}"
    image_alt += f". Background: {post.image_alt}" if post.image_alt else ". Social preview."
    tags = [START]
    tags.append(f'    <link rel="canonical" href="{escape(post.url, quote=True)}">')
    values = []
    if not post.authored_description:
        values.append(("name", "description", post.description))
    if not post.authored_author:
        values.append(("name", "author", post.author))
    values += [("property", "og:type", "article"),
               ("property", "og:site_name", "not so sparse"),
               ("property", "og:title", post.title),
               ("property", "og:description", post.description),
               ("property", "og:url", post.url),
               ("property", "og:image", image_url),
               ("property", "og:image:type", "image/jpeg"),
               ("property", "og:image:width", str(WIDTH)),
               ("property", "og:image:height", str(HEIGHT)),
               ("property", "og:image:alt", image_alt),
               ("name", "twitter:card", "summary_large_image"),
               ("name", "twitter:title", post.title),
               ("name", "twitter:description", post.description),
               ("name", "twitter:image", image_url),
               ("name", "twitter:image:alt", image_alt)]
    for attr, key, value in values:
        tags.append(f'    <meta {attr}="{key}" content="{escape(value, quote=True)}">')
    tags.append(END)
    return "\n".join(tags) + "\n"


def generate(root, check=False):
    root = root.resolve()
    posts, domain = read_posts(root)
    outputs = []
    # Validate/render everything before writing, so an invalid post stops the build.
    for post in posts:
        data = render_card(root, post, domain)
        target = root / "assets/social" / f"{post.path.parent.name}.jpg"
        image_url = f"https://{domain}/assets/social/{target.name}?v={sha256(data).hexdigest()[:12]}"
        page = re.sub(r"</head\s*>", lambda m: metadata(post, image_url) + m.group(),
                      post.source, count=1, flags=re.I)
        outputs.extend(((target, data), (post.path, page.encode("utf-8"))))
    changed = []
    for path, content in outputs:
        if not path.exists() or path.read_bytes() != content:
            changed.append(path.relative_to(root))
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    if check and changed:
        for path in changed:
            print(f"Out of date: {path}", file=sys.stderr)
        return 1
    print(f"{'Checked' if check else 'Generated'} {len(posts)} social cards; "
          f"{len(changed)} files {'need updating' if check else 'updated'}.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail on stale/missing generated files without writing")
    args = parser.parse_args()
    try:
        return generate(ROOT, args.check)
    except (OSError, ValueError, KeyError) as error:
        print(f"Social preview generation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
