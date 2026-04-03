# Deploy JobGuard Portal to GitHub

## Step 1: Create the repo on GitHub

1. Go to **https://github.com/new**
2. Repository name: `jobguard-ai`
3. Description: `Free fake job posting detector — REAL or FAKE verdict`
4. Choose **Public**
5. **Do NOT** add README, .gitignore, or license (we already have them)
6. Click **Create repository**

## Step 2: Push your code

```bash
cd /Users/santhakumar/Desktop/jobguard-ai

# Set your GitHub username (replace YOUR_USERNAME with your actual username)
git remote remove origin 2>/dev/null
git remote add origin https://github.com/YOUR_USERNAME/jobguard-ai.git

git push -u origin main
```

## Step 3: Enable GitHub Pages

1. In your repo, go to **Settings** → **Pages**
2. Under **Source**, select **Deploy from a branch**
3. Set **Branch** to `gh-pages`
4. Set **Folder** to `/ (root)`
5. Save

## Step 4: Wait for deploy

The workflow runs automatically. In 1–2 minutes your portal will be live at:

**https://YOUR_USERNAME.github.io/jobguard-ai**
