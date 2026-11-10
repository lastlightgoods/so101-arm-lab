"""Make LeRobot 0.6.x importable on Python 3.10 (Jetson). Rewrites the PEP 695 syntax 3.10 cannot parse:
  type X = ...                      ->  X = ...
  class C[T, U](Base):              ->  class C(Base, Generic[T, U]):      (+ TypeVars)
  def f[T: Bound](...):             ->  def f(...):                        (+ TypeVar with bound)
Idempotent. Run inside the venv after (re)installing lerobot."""
import importlib.util, pathlib, re, sys
root = pathlib.Path(importlib.util.find_spec("lerobot").origin).parent
GEN = re.compile(r"^(\s*)(class|def) (\w+)\[([^\]]+)\](\([^)]*\))?(.*)$", re.M)
import typing
TYPING_IMPORT = re.compile(r"^from typing import (\([^)]*\)|[^\n]+)$", re.M)
def fix_typing(src):
    """Move names that this Python's typing lacks (Self, override, TypeAliasType, ...) to typing_extensions."""
    def sub(m):
        body = m.group(1).strip("()"); names = [n.strip() for n in body.replace("\n", ",").split(",") if n.strip()]
        missing = [n for n in names if not hasattr(typing, n.split(" as ")[0].strip())]
        if not missing: return m.group(0)
        keep = [n for n in names if n not in missing]
        return (f"from typing import {', '.join(keep)}\n" if keep else "") + f"from typing_extensions import {', '.join(missing)}"
    return TYPING_IMPORT.sub(sub, src)
def names_of(params): return [x.strip().split(":")[0].strip() for x in params.split(",") if x.strip()]
changed = 0
for py in root.rglob("*.py"):
    src = py.read_text(); new = fix_typing(re.sub(r"^(\s*)type (\w+) = (.+)$", r"\1\2 = \3", src, flags=re.M)); tvs = []
    def fix(m):
        indent, kind, name, params, bases, rest = m.groups(); tvs.extend(names_of(params)); names = ", ".join(names_of(params))
        if kind == "class": return f"{indent}class {name}({bases[1:-1] + ', ' if bases and bases != '()' else ''}_Generic310[{names}]){rest}"
        return f"{indent}def {name}{bases or '()'}{rest}"
    new = GEN.sub(fix, new)
    if tvs:  # TypeVars go at the top of the file (bounds dropped: runtime does not need them); never between a decorator and its class
        hdr = "from typing import Generic as _Generic310, TypeVar as _TypeVar310\n" + "".join(f"{t} = _TypeVar310('{t}')\n" for t in dict.fromkeys(tvs))
        lines = new.split("\n"); i = 0
        if lines and lines[0].startswith("#!"): i = 1
        for k, l in enumerate(lines):   # must come after any `from __future__ import ...`
            if l.startswith("from __future__"): i = k + 1
        new = "\n".join(lines[:i] + [hdr.rstrip("\n")] + lines[i:])
    if new != src:
        try: compile(new, str(py), "exec")
        except SyntaxError as e: print("still broken:", py.relative_to(root), e.lineno, e.msg); sys.exit(1)
        py.write_text(new); changed += 1; print("patched", py.relative_to(root))
for py in root.rglob("*.py"):
    try: compile(py.read_text(), str(py), "exec")
    except SyntaxError as e: print("UNPATCHED SYNTAX:", py.relative_to(root), e.lineno); sys.exit(1)
print(f"done: {changed} files patched, all files compile on", sys.version.split()[0])
