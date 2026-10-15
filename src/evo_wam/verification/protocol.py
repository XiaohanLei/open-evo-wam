"""Native prompt/parser selection; retain backbone-specific camera crops."""

import base64, hashlib, importlib, json
from pathlib import Path


class Protocol:
    def __init__(self, name):
        if name not in ("droid", "robotwin_dense"):
            raise ValueError("Unknown VLM protocol")
        self.name = name
        package = "evo_wam.verification.protocols." + name
        self.contract = importlib.import_module(package + ".contract")
        self.root = Path(self.contract.__file__).parent
        if name == "droid":
            self.builder = importlib.import_module(package + ".payload").build_payload
        else:
            self.prompts = json.loads((self.root / "prompts.json").read_text())
            self.rules = json.loads((self.root / "task_rules.json").read_text())
            self.focus = json.loads((self.root / "focus.json").read_text())

    def parse(self, response, kind, description=None, task=None, goal=None):
        if self.name == "droid":
            return self.contract.parse(response, kind, description, task, goal)
        return self.contract.parse(response, kind, description)

    def payload(self, task, goal, kind, instruction, raw, description=None):
        hashes = [hashlib.sha256(x).hexdigest() for x in raw]
        if self.name == "droid":
            return self.builder(task, goal, kind, instruction, raw, hashes, description)
        if len(raw) != 6:
            raise ValueError("Six native views required")
        text = (
            "OBJECTS AND OBSERVATION DIMENSIONS: " + self.focus[task]
            if kind == "description"
            else "TASK INSTRUCTION: "
            + instruction
            + "\nSUCCESS RULE: "
            + self.rules[task]
            + "\nFALLIBLE DESCRIPTION: "
            + json.dumps(description, ensure_ascii=False)
        )
        content = [dict(type="text", text=text)]
        labels = [
            "INITIAL head",
            "INITIAL wrist left panel",
            "INITIAL wrist right panel",
            "CURRENT head",
            "CURRENT wrist left panel",
            "CURRENT wrist right panel",
        ]
        for label, image in zip(labels, raw):
            content.extend(
                [
                    dict(type="text", text=label),
                    dict(
                        type="image_url",
                        image_url=dict(
                            url="data:image/png;base64,"
                            + base64.b64encode(image).decode(),
                            detail="high",
                        ),
                    ),
                ]
            )
        return dict(
            model="vlm-local",
            messages=[
                dict(role="system", content=self.prompts[kind]),
                dict(role="user", content=content),
            ],
            max_tokens=12288,
            stream=False,
        )
