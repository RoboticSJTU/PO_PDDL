from __future__ import annotations

from collections.abc import Iterable, Mapping


def build_type_parent_map(rows: Iterable[Mapping[str, object]]) -> dict[str, str]:
    """Build the effective single-inheritance hierarchy rendered into PDDL.

    Concrete types can carry semantic memberships such as ``containable_item``
    in addition to their physical parent.  When all members of a semantic type
    share one physical parent, the renderer inserts that semantic type between
    the concrete type and the physical parent.  Observation grounding must use
    the same hierarchy or it will fail to recognize valid arguments.
    """
    normalized_rows = [row for row in rows if isinstance(row, Mapping)]
    parent_by_type = {
        str(row["type_name"]).strip(): str(row["parent_type"]).strip()
        for row in normalized_rows
        if str(row.get("type_name") or "").strip() and str(row.get("parent_type") or "").strip()
    }

    members_by_special_type: dict[str, list[str]] = {}
    for row in normalized_rows:
        concrete_type = str(row.get("type_name") or "").strip()
        if not concrete_type:
            continue
        memberships = row.get("special_supertypes") or []
        if not isinstance(memberships, list):
            continue
        for membership in memberships:
            special_type = str(membership).strip()
            if special_type and special_type != parent_by_type.get(concrete_type):
                members_by_special_type.setdefault(special_type, []).append(concrete_type)

    for special_type, member_types in sorted(members_by_special_type.items()):
        member_parents = {parent_by_type.get(member_type, "object") for member_type in member_types}
        if len(member_parents) != 1:
            continue
        common_parent = next(iter(member_parents))
        if common_parent in {"object", special_type}:
            continue
        parent_by_type[special_type] = common_parent
        for member_type in member_types:
            parent_by_type[member_type] = special_type

    return parent_by_type
