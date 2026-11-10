#!/usr/bin/env python3
"""cf_token.py · build (and optionally create) a least-privilege Cloudflare API token from permission NAMES.

    python cf_token.py --name so101-mlflow-tunnel --zone lastlightgoods.com \
        --account-perm "Cloudflare Tunnel Write" --account-perm "Access: Apps and Policies Write" \
        --zone-perm "DNS Write" --expires 2027-01-31
            -> prints the JSON body the API expects (dry run)
    ... --create         -> POSTs it and prints the new token's secret ONCE; store it, it cannot be read again

Needs a bootstrap token in CF_API_TOKEN (or ~/.config/lastlight/cloudflare.env, mode 600) that has "API Tokens Write"
(dashboard: User API Tokens: Edit) and, for --zone, Zone Read. Permission group ids are looked up live from the API, so the
names must match Cloudflare's API names exactly (they say Write where the dashboard says Edit).
Account id defaults to the one on the user's account list when there is exactly one."""
import argparse, json, os, sys, urllib.request, urllib.parse
API = "https://api.cloudflare.com/client/v4"
def token():
    t = os.environ.get("CF_API_TOKEN")
    if not t:
        try:
            for line in open(os.path.expanduser("~/.config/lastlight/cloudflare.env")):
                if line.startswith("CF_API_TOKEN="): t = line.split("=", 1)[1].strip().strip('"')
        except FileNotFoundError: pass
    if not t: sys.exit("no bootstrap token: set CF_API_TOKEN or put CF_API_TOKEN=... in ~/.config/lastlight/cloudflare.env")
    return t
def call(method, path, body=None):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body else None, method=method,
                                 headers={"Authorization": "Bearer " + token(), "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r: d = json.load(r)
    if not d.get("success"): sys.exit("API error: " + json.dumps(d.get("errors"), indent=1))
    return d["result"]
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True); ap.add_argument("--expires", help="YYYY-MM-DD (end of that day, UTC)")
    ap.add_argument("--account", help="account id; default: the only account the bootstrap token can see")
    ap.add_argument("--zone", help="zone name for the zone-scoped policy, e.g. lastlightgoods.com")
    ap.add_argument("--account-perm", action="append", default=[], help="API permission-group name, account scope (repeatable)")
    ap.add_argument("--zone-perm", action="append", default=[], help="API permission-group name, zone scope (repeatable)")
    ap.add_argument("--list", action="store_true", help="just list permission-group names and exit")
    ap.add_argument("--create", action="store_true", help="create the token (default is a dry run that prints the body)")
    a = ap.parse_args()
    groups = call("GET", "/user/tokens/permission_groups")
    if a.list:
        for g in sorted(groups, key=lambda g: (",".join(g.get("scopes", [])), g["name"])): print(f"{g['id']}  {','.join(s.split('.')[-1] for s in g.get('scopes', [])):10s}  {g['name']}")
        return
    by_name = {g["name"]: g for g in groups}
    def ids(names, scope):
        out = []
        for n in names:
            g = by_name.get(n)
            if not g: sys.exit(f"no permission group named {n!r}; run with --list to see the exact names")
            if scope and not any(s.endswith(scope) for s in g.get("scopes", [])): sys.exit(f"{n!r} is not a {scope}-scoped permission")
            out.append({"id": g["id"]})
        return out
    account = a.account
    if not account:
        accts = call("GET", "/accounts")
        if len(accts) != 1: sys.exit(f"pass --account; the token sees {len(accts)} accounts")
        account = accts[0]["id"]
    policies = []
    if a.account_perm: policies.append({"effect": "allow", "permission_groups": ids(a.account_perm, "account"), "resources": {f"com.cloudflare.api.account.{account}": "*"}})
    if a.zone_perm:
        if not a.zone: sys.exit("--zone-perm needs --zone")
        zones = call("GET", "/zones?" + urllib.parse.urlencode({"name": a.zone}))
        if not zones: sys.exit(f"zone {a.zone} not visible to the bootstrap token (needs Zone Read)")
        policies.append({"effect": "allow", "permission_groups": ids(a.zone_perm, "zone"), "resources": {f"com.cloudflare.api.account.zone.{zones[0]['id']}": "*"}})
    if not policies: sys.exit("nothing to grant: add --account-perm and/or --zone-perm")
    body = {"name": a.name, "policies": policies}
    if a.expires: body["expires_on"] = a.expires + "T23:59:59Z"
    if not a.create:
        print(json.dumps(body, indent=2)); print("\n(dry run; add --create to create it)", file=sys.stderr); return
    r = call("POST", "/user/tokens", body)
    print("created token id", r["id"]); print("SECRET (shown once, store it now):", r["value"])
if __name__ == "__main__": main()
