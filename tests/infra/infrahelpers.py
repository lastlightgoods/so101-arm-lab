"""Helpers for tests/infra: paths into infra/, a loader for docker-compose.yml, a .env parser and a Terraform regex scanner.

The compose file is read with PyYAML when it is installed (the Mac venv has it) and otherwise with ``load_yaml_subset``, a
40-line reader for the subset the compose file is written in: two-space block mappings, `- item` lists of scalars, flow lists
such as ["CMD", "python", "x.py"], quoted or bare scalars, comments. The file is deliberately kept inside that subset (no anchors,
no multi-line scalars, no `- key: value` list items) so CI, which installs only requirements-dev.txt, can parse it too.
When both readers are available, test_compose.py checks that they agree, which keeps the small reader honest."""
import os, re, subprocess
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INFRA = os.path.join(ROOT, "infra"); TF = os.path.join(INFRA, "terraform")
COMPOSE = os.path.join(INFRA, "docker-compose.yml"); ENV_EXAMPLE = os.path.join(INFRA, ".env.example")
DOCKERFILE = os.path.join(INFRA, "Dockerfile.mlflow"); ENTRYPOINT = os.path.join(INFRA, "entrypoint.sh")
MLFLOW_UI = os.path.join(ROOT, "tools", "mlflow_ui.sh"); MIGRATE = os.path.join(INFRA, "migrate_sqlite.sh")
# why: `$${X}` in a compose healthcheck is a literal `${X}` for the container's shell, not a compose variable
COMPOSE_VAR = re.compile(r"(?<!\$)\$\{([A-Z][A-Z0-9_]*)\}")
# the settings both builds share: compose reads them from .env as UPPER_CASE, Terraform declares them as lower_case variables
SHARED = ("MLFLOW_VERSION", "MLFLOW_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD", "MLFLOW_ARTIFACT_BUCKET")


def read(path): return open(path, encoding="utf-8").read()


def _split_flow(s):
    out, cur, q = [], "", None
    for ch in s:
        if q: cur += ch; q = None if ch == q else q
        elif ch in "\"'": q = ch; cur += ch
        elif ch == ",": out.append(cur); cur = ""
        else: cur += ch
    if cur.strip(): out.append(cur)
    return out


def _scalar(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'": return s[1:-1]
    if s.startswith("[") and s.endswith("]"): return [_scalar(x) for x in _split_flow(s[1:-1])]
    return int(s) if re.fullmatch(r"-?\d+", s) else s   # why: PyYAML reads a bare `retries: 12` as an int; agree with it


def load_yaml_subset(text):
    lines = [(len(l) - len(l.lstrip(" ")), l.split(" #")[0].strip() if not l.strip().startswith("#") else "") for l in text.splitlines()]
    lines = [(i, s) for i, s in lines if s]
    pos = 0

    def parse(indent):
        nonlocal pos
        if lines[pos][1].startswith("- "):
            out = []
            while pos < len(lines) and lines[pos][0] == indent and lines[pos][1].startswith("- "): out.append(_scalar(lines[pos][1][2:])); pos += 1
            return out
        out = {}
        while pos < len(lines) and lines[pos][0] == indent:
            key, _, val = lines[pos][1].partition(":"); pos += 1; key = _scalar(key)
            if val.strip(): out[key] = _scalar(val)
            elif pos < len(lines) and lines[pos][0] > indent: out[key] = parse(lines[pos][0])
            else: out[key] = None
        return out
    return parse(0)


def load_compose():
    text = read(COMPOSE)
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        return load_yaml_subset(text)


def env_keys(path=ENV_EXAMPLE):
    """Ordered {KEY: value} of the KEY=value lines in a .env file."""
    return dict(m.groups() for m in re.finditer(r"^([A-Z][A-Z0-9_]*)=(.*)$", read(path), re.M))


def tf_code(text):
    """Terraform text with the comments removed, for assertions about what is configured rather than what is explained."""
    return "\n".join(l.split("#")[0] for l in text.splitlines())


def tf_text(): return "\n".join(read(os.path.join(TF, f)) for f in sorted(os.listdir(TF)) if f.endswith(".tf"))


def tf_variables(): return set(re.findall(r'^variable\s+"([a-z0-9_]+)"', read(os.path.join(TF, "variables.tf")), re.M))


def tf_blocks(kind, text=None):
    """Bodies of every `kind { ... }` block (one brace level deep is enough for ingress/egress)."""
    return re.findall(kind + r"\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text or tf_text())


def sh_syntax_ok(path):
    return subprocess.run(["bash", "-n", path], capture_output=True, text=True)


@pytest.fixture(scope="session")
def compose(): return load_compose()


@pytest.fixture(scope="session")
def services(compose): return compose["services"]
