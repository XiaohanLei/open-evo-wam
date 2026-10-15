"""Frozen progressive thresholds and deletion-only historical prefix selection."""

import collections
import math
from evo_wam.io import read, sha


def threshold_record(schedule_path, scores_path, number):
    manifest = read(schedule_path)
    if sha(scores_path) != manifest["snapshot_sha256"]:
        raise ValueError("Calibration snapshot changed")
    schedule = manifest["schedule"]
    if (
        len(schedule) != 37
        or any(
            row["round"] != i
            or row["percentile"] != 90 - 2.5 * (i - 1)
            or not math.isfinite(row["threshold"])
            or row["threshold"] < 0
            for i, row in enumerate(schedule, 1)
        )
        or any(a["threshold"] < b["threshold"] for a, b in zip(schedule, schedule[1:]))
    ):
        raise ValueError("Invalid progressive schedule")
    if type(number) is not int or not 1 <= number <= 37:
        raise ValueError("Undefined progressive round; no automatic percentile floor")
    return dict(schedule[number - 1])


def select(number, pools, previous, limit):
    if type(number) is not int or number < 1 or not math.isfinite(limit) or limit < 0:
        raise ValueError("Invalid progressive selection boundary")
    prior = {r["source_logical_slot_id"]: r for r in previous}
    if len(prior) != len(previous):
        raise ValueError("Duplicate previous prefix")
    seen, merged, ledger, new_ids = set(), [], [], set()
    counts = collections.Counter()
    removed = 0
    for source_round, rows in pools:
        if not 1 <= source_round <= number:
            raise ValueError("Invalid source round")
        for row in rows:
            uid, score = row["source_logical_slot_id"], row["idm_mse"]
            if (
                uid in seen
                or row["round"] != source_round
                or not math.isfinite(score)
                or score < 0
            ):
                raise ValueError("Duplicate or invalid original prefix")
            seen.add(uid)
            old = prior.get(uid)
            if old is not None and (
                source_round >= number
                or {k: v for k, v in old.items() if k != "episode_index"}
                != {k: v for k, v in row.items() if k != "episode_index"}
            ):
                raise ValueError("Historical endpoint or provenance changed")
            is_new = source_round == number
            if is_new:
                new_ids.add(uid)
                if old is not None or score > limit:
                    raise ValueError("New prefix must pass current threshold")
            eligible = score <= limit
            if not is_new and old is None and eligible:
                raise ValueError("Previously removed prefix cannot re-enter")
            keep = eligible and (is_new or old is not None)
            deleted = old is not None and not keep
            removed += int(deleted)
            ledger.append(
                dict(
                    source_logical_slot_id=uid,
                    source_round=source_round,
                    accepted_step=row["accepted_step"],
                    idm_mse=score,
                    training_round=number,
                    threshold=limit,
                    retained=keep,
                    previously_active=old is not None,
                    removed_this_round=deleted,
                    reason="new_accepted"
                    if is_new
                    else "history_retained"
                    if keep
                    else "history_above_threshold"
                    if deleted
                    else "previously_removed",
                    **{
                        k: row[k]
                        for k in ("source_actions_sha256", "verifier_moment_sha256")
                        if k in row
                    },
                )
            )
            if keep:
                merged.append(dict(row, episode_index=len(merged)))
                counts[str(source_round)] += 1
    if not set(prior).issubset(seen):
        raise ValueError("Missing original history")
    if not new_ids or not merged:
        raise ValueError("No newly accepted prefixes; do not weaken the threshold")
    return (
        merged,
        ledger,
        dict(
            episodes=len(merged),
            new_accepted=len(new_ids),
            previous_active=len(previous),
            historical_removed=removed,
            historical_retained=len(previous) - removed,
            previously_removed=sum(r["reason"] == "previously_removed" for r in ledger),
            retained_by_source_round=dict(counts),
            task_config_counts=dict(
                collections.Counter(r["task"] + "/" + r["split"] for r in merged)
            ),
            prefix_steps=dict(
                min=min(r["accepted_step"] for r in merged),
                max=max(r["accepted_step"] for r in merged),
                mean=sum(r["accepted_step"] for r in merged) / len(merged),
            ),
            retained_action_steps=sum(r["accepted_step"] for r in merged),
        ),
    )
