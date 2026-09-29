# Academic site

Static HTML for [aniketdeshpande.com](https://aniketdeshpande.com/), published with GitHub Pages and its existing Jekyll configuration.

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
