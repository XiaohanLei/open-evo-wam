"""Exact endpoint scoring. Window ranges are half open action indices."""

import hashlib, json, math

VERSION = "full_prefix_v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def layout(length):
    if type(length) is not int or not 1 <= length <= 64:
        raise ValueError("Expected 1 <= L <= 64")
    future = (length + 3) // 4
    return dict(
        length=length,
        input_frames=length + 1,
        encoded_frames=4 * future + 1,
        repeat_last_frames=4 * future - length,
        future_latents=future,
    )


def ranges(t):
    if type(t) is not int or t < 1:
        raise ValueError("Endpoint must be a positive action count")
    return [(s, min(s + 64, t)) for s in range(0, t, 64)]


def required_windows(endpoints):
    return sorted({pair for t in endpoints for pair in ranges(t)})


def select_windows(records, t):
    wanted = ranges(t)
    found = {}
    for rec in records:
        pair = (rec["start"], rec["end"])
        if pair not in wanted:
            continue
        if pair in found:
            raise ValueError("Duplicate IDM window")
        if rec.get("scoring_version") != VERSION:
            raise ValueError("Old scoring cache is incompatible")
        length = pair[1] - pair[0]
        if rec.get("length") != length or rec.get("k") != pair[0] // 64:
            raise ValueError("Invalid window range")
        if rec.get("candidate_action_conditioning") is not False:
            raise ValueError("Candidate-conditioned reconstruction")
        score = rec["normalized_mse"]
        if not math.isfinite(score) or score < 0:
            raise ValueError("Nonfinite or negative score")
        found[pair] = rec
    if set(found) != set(wanted):
        raise ValueError("Missing exact endpoint IDM windows")
    return [found[pair] for pair in wanted]


def prefix_score(records, t):
    return (
        sum(r["length"] * r["normalized_mse"] for r in select_windows(records, t)) / t
    )


def summary(records, t):
    selected = select_windows(records, t)
    return dict(
        scoring_version=VERSION,
        accepted_step=t,
        scored_actions=t,
        unscored_tail_actions=0,
        normalized_mse=prefix_score(records, t),
        window_ranges=[[r["start"], r["end"]] for r in selected],
    )
