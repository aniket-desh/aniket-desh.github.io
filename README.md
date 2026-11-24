# Academic site

Static HTML for [aniketdeshpande.com](https://aniketdeshpande.com/), published with GitHub Pages and its existing Jekyll configuration.

## Layout

Every page is plain HTML with no build-time templating. Code shared between pages lives in `assets/`; code that belongs to one post lives next to that post.

- `assets/post.css` holds the defaults every post starts from, and `assets/post.js` handles footnotes and table-of-contents highlighting.
- `assets/reading-theme.css`, `assets/reading-theme.js` and `assets/reading.css` hold the light and dark palettes and the reading layout.
- `blog/<slug>/post.css` holds that post's own styles, and any other `.js` or `.css` in the folder is that post's figures. Drosophila is a daily problem log: each day lives at `drosophila/YYYY-MM-DD/` with the problem's first sentence as the `h1`, the rest in `.problem-statement`, the date in `.post-meta`, and the solution in `.post-body`; all drosophila pages share `drosophila/notes.css` in the same role.
- `assets/listing.css` and `assets/listing.js` serve the two listing pages, `/blog/` and `/drosophila/`.

A post's `<head>` loads its stylesheets in this order, and the order matters because later files override earlier ones:

```html
<link rel="stylesheet" href="/assets/post.css">
<link rel="stylesheet" href="/blog/<slug>/post.css">
<link rel="stylesheet" href="/assets/reading-theme.css">
<link rel="stylesheet" href="/assets/reading.css">
```

`<script src="/assets/post.js"></script>` goes at the end of `<body>`, after the post's own scripts. Footnotes are either keyed (`<sup data-fn="key"></sup>` with `<li data-fn="key">`) or numbered (`<sup>3</sup>` for the third list item); the script picks the mode from the list. `data-toc-offset` and `data-toc-default="first"` on `.toc-sidebar` adjust when a section counts as reached.

## Blog sharing previews

`scripts/generate_social_cards.py` creates a 1200 × 630 JPEG for each published blog post and adds Open Graph and large Twitter Card metadata to its HTML. The image uses the post's existing hero, a dark gradient, its title, and the author's name in the site's serif typography. The article's visible content and hero remain unchanged.

Posts are discovered from live `.post-entry .read-more` links in `blog/index.html`; commented entries and unlisted drafts are excluded. A new post needs the same structure as the existing posts:

- `h1.post-header` supplies its title.
- `.post-subtitle` supplies the description when there is no existing authored description.
- `.post-meta` supplies the author.
- `.hero-image img` supplies its local hero image.

Existing authored `<meta name="description">` values are preserved and reused. Generated images live in `assets/social/`; generated metadata is maintained inside marked blocks in each post's `<head>`. Edit the post's source fields, then regenerate instead of editing those blocks by hand.

To generate and check locally from the repository root, use Python 3.11 or later (CI uses Python 3.12):

```sh
python3 -m venv /tmp/academic-site-social-venv
source /tmp/academic-site-social-venv/bin/activate
python -m pip install -r scripts/requirements-social.txt
python scripts/generate_social_cards.py
python scripts/generate_social_cards.py --check
python -m http.server 8765 --bind 127.0.0.1
```

Open the site at `http://127.0.0.1:8765/` and inspect the JPEGs under `/assets/social/`. Generated HTML and images can be committed for convenient local previews; the deployment workflow regenerates them from the current post content.

## Automatic generation when publishing

`.github/workflows/pages.yml` generates and checks previews on every push to `main`, builds with `actions/jekyll-build-pages`, then uploads and deploys the resulting site. Generation happens in the build workspace; CI does **not** commit generated files back to the repository. This avoids relying on bot commits, which [do not trigger a branch-based Pages build when pushed with `GITHUB_TOKEN`](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

The repository uses **GitHub Actions** as its Pages source. For a new clone deployed to a different repository, select **Settings → Pages → Build and deployment → Source → GitHub Actions**, then run **Build and deploy site** or push to `main`. GitHub documents this setup in [Using custom workflows with GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

Once activated, publishing a post and linking it from the blog index is enough to generate its sharing card during deployment. X controls the final presentation and may cache an older card after an update.
