from pathlib import Path
from operator import methodcaller

def ls(path: Path, infixes: str|list[str]|None = None,
       prefixes: str|list[str]|None = None,
       suffixes: str|list[str]|None = None,
       names_only: bool=False
       )-> list[Path] | list[str]:
    """List out the objects in a path filtered by infixes, prefixes, or suffixes."""

    def normalize(val: str | list[str] | None) -> tuple[str, ...]:
        if not val:
            return ()
        return (val,) if isinstance(val, str) else tuple(val)

    def keep_path(p: Path) -> bool:
        name = p.name
        inf, pre, suf = (
            normalize(infixes),
            normalize(prefixes),
            normalize(suffixes),
        )

        # Check infixes (contains at least one)
        if inf and not any(i in name for i in inf):
            return False
        # Check prefixes (starts with at least one)
        if pre and not name.startswith(pre):
            return False
        # Check suffixes (ends with at least one)
        if suf and not name.endswith(suf):
            return False

        return True

    results = [p for p in path.iterdir() if keep_path(p)]

    if names_only:
        return [p.name for p in results]
    return results