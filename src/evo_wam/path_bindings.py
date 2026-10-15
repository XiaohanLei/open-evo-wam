"""Explicit relocation of manifest references without modifying frozen evidence."""

from pathlib import PurePosixPath


class PathBindings:
    def __init__(self, *groups):
        self.mapping = {}
        for group in groups:
            for item in group or []:
                if set(item) != {"from", "to"}:
                    raise ValueError("Path bindings require exactly from and to")
                old, new = item["from"], item["to"]
                for path in (old, new):
                    if (
                        not isinstance(path, str)
                        or not path.startswith("/")
                        or ".." in PurePosixPath(path).parts
                    ):
                        raise ValueError(
                            "Path bindings require absolute paths without traversal"
                        )
                old, new = str(PurePosixPath(old)), str(PurePosixPath(new))
                if old == "/" or new == "/":
                    raise ValueError("A filesystem root cannot be rebound")
                if old in self.mapping and self.mapping[old] != new:
                    raise ValueError("Conflicting path bindings for " + old)
                self.mapping[old] = new
        # A serialized, already relocated pool must remain unchanged on reread.
        for destination in self.mapping.values():
            if any(
                destination == source or destination.startswith(source + "/")
                for source in self.mapping
            ):
                raise ValueError("Chained or cyclic path bindings are not supported")

    def path(self, value):
        if not isinstance(value, str) or not value.startswith("/"):
            return value
        if ".." in PurePosixPath(value).parts:
            raise ValueError("Manifest reference contains path traversal")
        value = str(PurePosixPath(value))
        for source in sorted(self.mapping, key=len, reverse=True):
            if value == source or value.startswith(source + "/"):
                return self.mapping[source] + value[len(source) :]
        return value

    def references(self, value, key=""):
        if isinstance(value, dict):
            return {k: self.references(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [self.references(v, key) for v in value]
        return self.path(value) if key in {"path", "case"} else value

    def records(self):
        return [{"from": old, "to": new} for old, new in sorted(self.mapping.items())]
