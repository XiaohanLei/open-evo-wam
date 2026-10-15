"""Chronological endpoint selection and fixed-endpoint vote3, independent of I/O."""


def scan(goals, prepare, chain, *, vote_timing="after_scan"):
    if vote_timing not in ("after_scan", "per_goal", "per_moment"):
        raise ValueError("Invalid vote timing")
    history = []
    anchors = []
    previous = 0

    def vote(anchor):
        votes = [anchor["vote0"]] + [
            chain(anchor["goal"], anchor["step"], i, anchor["evidence"]) for i in (1, 2)
        ]
        return dict(
            goal=anchor["goal"],
            endpoint=anchor["step"],
            votes=[
                dict(
                    id=str(i),
                    endpoint=anchor["step"],
                    accepted=v["decision"] == "accept",
                    record=v,
                )
                for i, v in enumerate(votes)
            ],
        )

    groups = []
    for goal, points in enumerate(goals):
        if any(type(t) is not int or t < 1 for t in points) or points != sorted(
            set(points)
        ):
            raise ValueError("Scan points must be positive unique ordered integers")
        anchor = None
        for step in points:
            if step <= previous:
                continue
            evidence = prepare(goal, step)
            if evidence.get("action_filtered"):
                return dict(
                    accepted=False,
                    reason="action_filtered",
                    history=history,
                    vote_groups=groups,
                )
            if evidence.get("gate_skipped"):
                history.append(dict(goal=goal, step=step, decision="idm_filtered"))
                continue
            v0 = chain(goal, step, 0, evidence)
            if v0.get("decision") not in ("accept", "reject", "uncertain"):
                raise ValueError(
                    "Missing/invalid VLM decision cannot count as rejection"
                )
            history.append(dict(goal=goal, step=step, decision=v0["decision"]))
            if v0["decision"] == "accept":
                anchor = dict(goal=goal, step=step, evidence=evidence, vote0=v0)
                if vote_timing == "per_moment":
                    group = vote(anchor)
                    if sum(v["accepted"] for v in group["votes"]) < 2:
                        history[-1]["decision"] = "fixed_endpoint_vote_rejected"
                        anchor = None
                        continue
                    groups.append(group)
                anchors.append(anchor)
                previous = step
                break
        if anchor is None:
            return dict(
                accepted=False,
                reason="subgoal_unconfirmed",
                history=history,
                vote_groups=groups,
            )
        if vote_timing == "per_goal":
            group = vote(anchor)
            groups.append(group)
            if sum(v["accepted"] for v in group["votes"]) < 2:
                return dict(
                    accepted=False,
                    reason="fixed_endpoint_vote_rejected",
                    history=history,
                    vote_groups=groups,
                )
    if vote_timing == "after_scan":
        groups = [vote(a) for a in anchors]
    accepted = all(sum(v["accepted"] for v in g["votes"]) >= 2 for g in groups)
    return dict(
        accepted=accepted,
        reason="accepted" if accepted else "fixed_endpoint_vote_rejected",
        visual_endpoint=previous,
        accepted_step=previous,
        history=history,
        vote_groups=groups,
    )
