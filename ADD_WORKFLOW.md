# Add GitHub Pages Workflow

Your code is pushed to https://github.com/Santhakumarramesh/jobguard-ai

To enable auto-deploy to GitHub Pages:

1. Go to **https://github.com/Santhakumarramesh/jobguard-ai**
2. Click **Actions** → **New workflow** → **set up a workflow yourself**
3. Name the file: `deploy.yml`
4. Replace the default content with:

```yaml
name: Deploy JobGuard AI to GitHub Pages

on:
  push:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Pages
        uses: actions/configure-pages@v4

      - name: Prepare site
        run: |
          mkdir -p _site
          cp index.html _site/
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: '_site'

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

5. Click **Commit changes**
6. Go to **Settings** → **Pages** → Source: **GitHub Actions**

Your portal will be live at: **https://santhakumarramesh.github.io/jobguard-ai**
