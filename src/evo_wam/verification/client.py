"""Recorded HTTP requests with bounded retries and strict cache identities."""

import json, os, time, urllib.request, urllib.error
from pathlib import Path
from evo_wam.io import digest, read, write


class Client:
    def __init__(
        self,
        root,
        endpoint,
        model_revision,
        parser,
        *,
        max_attempts=6,
        timeout=240,
        api_key_env=None,
        transport=None,
        retry_delay=1,
    ):
        if not model_revision:
            raise ValueError("An explicit model revision is required")
        if not 1 <= max_attempts <= 6:
            raise ValueError("max_attempts must be between 1 and 6")
        self.root = Path(root)
        self.endpoint = endpoint
        self.revision = model_revision
        self.parser = parser
        self.attempts = max_attempts
        self.timeout = timeout
        self.key_env = api_key_env
        self.transport = transport or self.http
        self.delay = retry_delay

    def http(self, payload):
        headers = {"Content-Type": "application/json"}
        if self.key_env:
            headers["Authorization"] = "Bearer " + os.environ[self.key_env]
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(payload).encode(), headers=headers
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def request(self, key, payload, parse_context):
        if Path(key).is_absolute() or ".." in Path(key).parts:
            raise ValueError("Unsafe request key")
        identity = digest(
            dict(
                payload=payload,
                endpoint=self.endpoint,
                model_revision=self.revision,
                parse_context=parse_context,
            )
        )
        directory = self.root / key
        canonical = directory / "result.json"
        if canonical.exists():
            old = read(canonical)
            if old["identity"] != identity:
                raise ValueError("VLM request identity changed")
            # Parse recorded bytes again; never trust a mutable cached prediction.
            prediction, error = self.parser(old["response"], **parse_context)
            if error or prediction != old["prediction"]:
                raise ValueError("VLM cached response changed")
            return old
        for attempt in range(self.attempts):
            submitted = directory / f"attempt{attempt}.submitted.json"
            target = directory / f"attempt{attempt}.json"
            if target.exists():
                record = read(target)
                if record["identity"] != identity:
                    raise ValueError("VLM attempt identity changed")
            elif submitted.exists():
                raise RuntimeError(
                    "VLM submission has no recorded response; reconcile before retry: "
                    + str(submitted)
                )
            else:
                write(
                    submitted,
                    dict(
                        identity=identity,
                        payload=payload,
                        model_revision=self.revision,
                        submitted=time.time(),
                    ),
                )
                record = dict(
                    identity=identity,
                    prediction=None,
                    error=None,
                    response=None,
                    retryable=True,
                )
                try:
                    record["response"] = self.transport(payload)
                    record["prediction"], record["error"] = self.parser(
                        record["response"], **parse_context
                    )
                except urllib.error.HTTPError as exc:
                    record.update(
                        error="HTTP " + str(exc.code),
                        retryable=exc.code == 429 or exc.code >= 500,
                    )
                except (TimeoutError, urllib.error.URLError, OSError) as exc:
                    record["error"] = type(exc).__name__
                except Exception as exc:
                    record.update(error=type(exc).__name__, retryable=False)
                write(target, record)
            if record["prediction"] is not None:
                parsed, error = self.parser(record["response"], **parse_context)
                if error or parsed != record["prediction"]:
                    raise ValueError("VLM recorded attempt changed")
                write(canonical, record)
                return record
            if not record["retryable"]:
                break
            if attempt + 1 < self.attempts and self.delay:
                time.sleep(self.delay)
        raise RuntimeError("VLM request exhausted or failed: " + str(directory))
