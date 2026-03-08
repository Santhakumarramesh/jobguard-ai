# Enable GitHub Pages — Fix 404

Your `gh-pages` branch has the portal files. Enable Pages in repo settings:

## Steps

1. Go to **https://github.com/Santhakumarramesh/jobguard-ai**
2. Click **Settings** → **Pages** (left sidebar)
3. Under **Build and deployment**:
   - **Source:** Deploy from a branch
   - **Branch:** `gh-pages` (select from dropdown)
   - **Folder:** `/ (root)`
4. Click **Save**

## Wait 1–2 minutes

Your portal will be live at:

**https://santhakumarramesh.github.io/jobguard-ai**

---

If the branch dropdown doesn't show `gh-pages`, the workflow may not have run. Go to **Actions** and run the workflow manually (Run workflow).
