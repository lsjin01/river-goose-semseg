"""Seven-class taxonomy that keeps source classes but merges all algae labels."""

CLASS_NAMES = (
    "river",
    "land",
    "bridge",
    "other",
    "nps",
    "turbid",
    "algae_including_nps_algae",
)

ALGAE_SOURCE_NAMES = frozenset(
    {"algae", "algae0", "algae1", "algae2", "algae3", "algae4", "nps_algae"}
)
IGNORE_SOURCE_NAMES = frozenset({"ambiguous"})
SOURCE_TO_CLASS_ID = {
    "river": 0,
    "land": 1,
    "bridge": 2,
    "other": 3,
    "nps": 4,
    "turbid": 5,
    **{name: 6 for name in ALGAE_SOURCE_NAMES},
}


def source_name_to_label(name: str) -> int:
    normalized = name.strip().lower()
    if normalized in IGNORE_SOURCE_NAMES:
        return 255
    try:
        return SOURCE_TO_CLASS_ID[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown source category name: {name!r}") from exc


def taxonomy_metadata() -> dict:
    return {
        "name": "merged_algae_7class",
        "class_names": list(CLASS_NAMES),
        "source_to_class_id": dict(sorted(SOURCE_TO_CLASS_ID.items())),
        "merged_algae_source_names": sorted(ALGAE_SOURCE_NAMES),
        "ignore_source_names": sorted(IGNORE_SOURCE_NAMES),
        "ignore_index": 255,
    }
