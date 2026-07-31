# Deploying to Render (free tier, no credit card)

This repo now has `Dockerfile` + `render.yaml`, so Render can auto-detect
everything. Steps you need to do yourself (Render's GitHub OAuth connection
can't be driven by anyone but you):

1. Go to https://render.com and sign up (no credit card required for the free
   tier as of 2026).
2. Dashboard → **New** → **Blueprint**.
3. Connect your GitHub account, then select `mateho-369/Auto-Clip-Engine`.
4. Render detects `render.yaml` automatically and shows the `auto-clip-engine`
   web service on the **Free** plan. Click **Apply** / **Deploy Blueprint**.
5. First build takes a few minutes (installing ffmpeg + Python deps). After
   that, your app is live at `https://auto-clip-engine.onrender.com` (or
   whatever subdomain Render assigns).

## This IS your CI/CD
Once connected, Render watches the `main` branch: every `git push` triggers
an automatic rebuild + redeploy. No separate GitHub Actions workflow needed —
Render's Git integration *is* the CI/CD pipeline here.

## Known limitation (free tier)
The free web service **spins down after 15 minutes of inactivity** and takes
30-60 seconds to wake up on the next request (cold start). Fine for a personal
tool/demo; if you later need it always-on and responsive, that requires a paid
plan (Starter, ~$7/mo).

## Custom domain (optional)
If you want `auto-clip-engine.great-site.net` (or a domain you actually
control the DNS for) pointing at this instead of InfinityFree: Render
dashboard → your service → **Settings** → **Custom Domains** → add the domain,
then add the CNAME record Render gives you at your DNS provider. This only
works for domains where you control DNS — InfinityFree's free subdomains do
not allow custom CNAME records pointing off-platform, so this specific
`great-site.net` domain can't be repointed this way.
