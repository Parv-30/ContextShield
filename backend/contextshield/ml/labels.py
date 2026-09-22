LABELS = ["hate_speech", "offensive", "threat", "harassment", "none"]


def toxicity_score(label_probs: dict[str, float]) -> float:
    non_none = [label_probs[label] for label in LABELS if label != "none"]
    return max(non_none) if non_none else 0.0


def map_hasoc_label(task1: str, task2: str | None) -> list[str]:
    if task1 == "NOT":
        return ["none"]
    mapping = {"HATE": "hate_speech", "OFFN": "offensive", "PRFN": "offensive"}
    return [mapping.get(task2, "offensive")]


def map_dravidian_label(raw_label: str) -> list[str]:
    mapping = {
        "Not_offensive": ["none"],
        "not-Tamil": ["none"],
        "Offensive_Untargeted": ["offensive"],
        "Offensive_Targeted_Insult_Individual": ["harassment"],
        "Offensive_Targeted_Insult_Group": ["hate_speech"],
        "Offensive_Targeted_Insult_Other": ["offensive"],
    }
    return mapping.get(raw_label, ["none"])
