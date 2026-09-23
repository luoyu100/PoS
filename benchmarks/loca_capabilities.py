"""Declarative LOCA tool capabilities used by PoS decision views.

This is an adapter profile, not a name-based classifier. Every entry is an
explicit contract for one stable benchmark tool interface; unlisted tools
remain semantically unknown to PoS.
"""

from __future__ import annotations

from copy import deepcopy


LOCA_TOOL_CAPABILITIES = {
    "filesystem_read_file": {
        "effect": "read",
        "consumes": ["source_resource"],
        "provides": ["source_content"],
        "argument_roles": {"source_resource": "path"},
    },
    "filesystem_read_text_file": {
        "effect": "read",
        "consumes": ["source_resource"],
        "provides": ["source_content"],
        "argument_roles": {"source_resource": "path"},
    },
    "filesystem_read_multiple_files": {
        "effect": "read",
        "consumes": ["source_resource"],
        "provides": ["source_content"],
        "argument_roles": {"source_resource": "path"},
    },
    "python_execute_python_execute": {
        "effect": "transform",
        "consumes": ["source_content"],
        "provides": ["durable_workspace_output"],
    },
    "google_sheet_list_spreadsheets": {
        "effect": "inspect",
        "consumes": [],
        # A list-all catalog can support Agent discovery, but its many IDs are
        # not scoped to the current delivery destination. Only a target-scoped
        # provider may bind destination_id in the execution lifecycle.
        "provides": ["resource_catalog"],
    },
    "google_sheet_get_multiple_spreadsheet_summary": {
        "effect": "inspect",
        "consumes": ["destination_id"],
        "provides": ["source_schema"],
    },
    "google_sheet_get_sheet_data": {
        "effect": "read",
        "consumes": ["destination_id", "destination_locator"],
        "provides": ["source_content", "destination_content"],
        "argument_roles": {
            "destination_id": "spreadsheet_id",
            "destination_locator": "sheet",
        },
    },
    "google_sheet_list_sheets": {
        "effect": "inspect",
        "consumes": ["destination_id"],
        "provides": ["destination_locator"],
        "argument_roles": {"destination_id": "spreadsheet_id"},
        "output_roles": {"destination_locator": "$items"},
    },
    "google_sheet_create_spreadsheet": {
        "effect": "create",
        "consumes": [],
        # Creating a spreadsheet returns its ID, but not an observed sheet/tab
        # locator. A separate schema/discovery call must bind that value.
        "provides": ["destination_id"],
        "output_roles": {"destination_id": ["spreadsheetId", "spreadsheet_id"]},
    },
    "google_sheet_update_cells": {
        "effect": "write",
        "consumes": [
            "destination_id",
            "destination_locator",
            "source_content",
        ],
        "provides": ["write_receipt"],
        "argument_roles": {
            "destination_id": "spreadsheet_id",
            "destination_locator": "sheet",
            "source_content": "data",
        },
        # update_cells requires an exact A1 range in addition to its tabular
        # data. This interface argument is derived from the observed 2D shape.
        "argument_derivations": {
            "range": {
                "source_role": "source_content",
                "operator": "a1_extent",
            }
        },
    },
    "google_sheet_batch_update_cells": {
        "effect": "write",
        "consumes": [
            "destination_id",
            "destination_locator",
            "source_content",
        ],
        "provides": ["write_receipt"],
        "argument_roles": {
            "destination_id": "spreadsheet_id",
            "destination_locator": "sheet",
            "source_content": "ranges",
        },
    },
    "claim_done_claim_done": {
        "effect": "finish",
        "consumes": ["validation_evidence"],
        "provides": ["environment_confirmation"],
    },
}


def attach_loca_capabilities(actions: list[dict]) -> list[dict]:
    """Attach only explicitly declared capabilities to current case actions."""

    for action in actions:
        function = action.get("function", {}) if isinstance(action, dict) else {}
        if not isinstance(function, dict):
            continue
        capability = LOCA_TOOL_CAPABILITIES.get(str(function.get("name", "")))
        if capability is not None:
            function["x_pos_capability"] = deepcopy(capability)
    return actions
