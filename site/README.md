# so101 arm lab — static site

This folder is the public website for the so101 arm lab. Everything in `public/` is served as-is by Cloudflare Workers static assets (no Worker script, no bindings, no KV); `wrangler.jsonc` holds the config, project name `so101-arm-lab`.

**Live URL:** https://so101-arm-lab.c-thomas-clark-pdx.workers.dev

**Deploy:** edit files in `public/`, then run `./deploy.sh` (equivalent to `cd site && wrangler deploy`). Uploads only changed files; takes a few seconds.

**Local preview:** `wrangler dev` from this folder, then open the printed localhost URL.

**Custom domain later (e.g. lab.lastlightgoods.com):** the zone must be on Cloudflare DNS. Then either add `"routes": [{ "pattern": "lab.lastlightgoods.com", "custom_domain": true }]` to `wrangler.jsonc` and redeploy (Cloudflare creates the DNS record and cert automatically), or in the dashboard go to Workers & Pages > so101-arm-lab > Settings > Domains & Routes > Add > Custom Domain. Not done yet; nothing here touches DNS.

**Ignore:** `.gitignore` excludes `node_modules/`, `.wrangler/` (local cache), and `.dev.vars`.
