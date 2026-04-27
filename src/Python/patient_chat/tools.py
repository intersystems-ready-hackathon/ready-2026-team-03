"""Patient-admission tools, backed by IRIS through `intersystems-irispython`.

Each tool is a typed Python function decorated with LangChain's
:func:`langchain_core.tools.tool`. The decorator builds a Pydantic schema
from the function's signature + type hints + docstring, and
:func:`convert_to_openai_tool` turns that into the OpenAI tool-calling
schema the agent expects — so there is no JSON to maintain by hand.

Column metadata for `Data.Patients` is **discovered at runtime** by
querying ``INFORMATION_SCHEMA`` (see :func:`describe_table`), so adding /
renaming columns in `Data.Patients.cls` does not require touching this
file.
"""

from __future__ import annotations

import logging
import re
from contextlib import closing
from typing import Annotated, Literal

from langchain_core.tools import StructuredTool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from db import connect


logger = logging.getLogger(__name__)

# Schema-qualified table name. Split lazily so SCHEMA / NAME are derived
# from this single source of truth.
TABLE = "Data.Patients"
TABLE_SCHEMA, TABLE_NAME = TABLE.split(".", 1)

# Columns that are managed by the database / class definition, not by users.
_AUDIT_COLUMNS = frozenset({"ID", "CreatedAt", "UpdatedAt"})

# Columns that identify a patient — never updated through `update_patient`.
_KEY_COLUMNS = frozenset({"SSN"})

GenderLiteral = Literal["M", "F", "Other", "Unknown"]

# US Social Security Numbers are 9 digits. We accept input with optional
# dashes/spaces (e.g. "123-45-6789" or "123 45 6789") and store them as
# the raw 9 digits.
_SSN_DIGITS_RE = re.compile(r"^\d{9}$")
_SSN_DESCRIPTION = (
    "Social Security Number — exactly 9 digits. "
    "Dashes/spaces are accepted (e.g. '123-45-6789') but stripped before storage."
)


def _normalize_ssn(ssn: str) -> str | None:
    """Strip separators and return the 9-digit SSN, or None if invalid."""
    if not isinstance(ssn, str):
        return None
    cleaned = re.sub(r"[\s-]", "", ssn)
    return cleaned if _SSN_DIGITS_RE.fullmatch(cleaned) else None


def _ssn_error(ssn: str) -> dict:
    return {
        "status": "error",
        "error": (
            f"Invalid SSN '{ssn}'. SSN must be exactly 9 digits "
            "(dashes/spaces optional). Please ask the patient to repeat it."
        ),
    }


# ---------- introspection helpers (cached) ----------

_columns_cache: dict[tuple[str, str], list[str]] = {}


def _fetch_columns(table_schema: str, table_name: str) -> list[str]:
    """Return the column names of ``schema.table`` in declaration order."""
    sql = (
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? "
        "ORDER BY ORDINAL_POSITION"
    )
    with closing(connect()) as db, db.cursor() as cur:
        cur.execute(sql, [table_schema, table_name])
        return [r[0] for r in cur.fetchall()]


def _get_table_columns(
    table_schema: str = TABLE_SCHEMA,
    table_name: str = TABLE_NAME,
) -> list[str]:
    """Cached column lookup for ``schema.table``. Cache is in-process."""
    key = (table_schema, table_name)
    cols = _columns_cache.get(key)
    if cols is None:
        cols = _fetch_columns(table_schema, table_name)
        if not cols:
            raise RuntimeError(
                f"INFORMATION_SCHEMA returned no columns for {table_schema}.{table_name}. "
                "Make sure the persistent class is compiled in IRIS."
            )
        _columns_cache[key] = cols
        logger.info("Discovered columns for %s.%s: %s", table_schema, table_name, cols)
    return cols


def _patient_select_columns() -> list[str]:
    """Columns used in SELECT statements (everything except ID / audit)."""
    return [c for c in _get_table_columns() if c not in _AUDIT_COLUMNS]


def _patient_updatable_columns() -> list[str]:
    """Columns that `update_patient` is allowed to write to."""
    return [
        c for c in _get_table_columns()
        if c not in _AUDIT_COLUMNS and c not in _KEY_COLUMNS
    ]


def _row_to_dict(cursor, row) -> dict:
    cols = [d[0] for d in cursor.description]
    return {c: (str(v) if v is not None else None) for c, v in zip(cols, row)}


# ---------- introspection tools ----------

@tool
def get_tables(
    table_schema: Annotated[
        str, Field(description="The IRIS SQL schema name. Defaults to 'Data'.")
    ] = TABLE_SCHEMA,
) -> list[str]:
    """List all tables in a specific SQL schema."""
    sql = (
        "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_SCHEMA = ? ORDER BY TABLE_NAME"
    )
    with closing(connect()) as db, db.cursor() as cur:
        cur.execute(sql, [table_schema])
        names = [r[0] for r in cur.fetchall()]
    logger.info("get_tables(%s) -> %d table(s)", table_schema, len(names))
    return names


@tool
def describe_table(
    table_name: Annotated[str, Field(description="The SQL table name.")],
    table_schema: Annotated[
        str, Field(description="The IRIS SQL schema name. Defaults to 'Data'.")
    ] = TABLE_SCHEMA,
) -> list[dict]:
    """Show columns and SQL types for a table.

    Returns a list of ``{"name", "type", "nullable"}`` dicts, ordered by
    ordinal position.
    """
    sql = (
        "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE "
        "FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? "
        "ORDER BY ORDINAL_POSITION"
    )
    with closing(connect()) as db, db.cursor() as cur:
        cur.execute(sql, [table_schema, table_name])
        return [
            {"name": r[0], "type": r[1], "nullable": r[2]}
            for r in cur.fetchall()
        ]


# ---------- patient tools ----------

@tool
def find_patient_by_ssn(
    ssn: Annotated[str, Field(description=_SSN_DESCRIPTION)],
) -> dict | None:
    """Look up a patient by Social Security Number (SSN).

    Returns the patient record, or null if no patient with that SSN exists.
    Returns an error dict if the SSN is not a valid 9-digit number.
    """
    normalized = _normalize_ssn(ssn)
    if normalized is None:
        logger.warning("find_patient_by_ssn rejected invalid SSN=%r", ssn)
        return _ssn_error(ssn)

    cols = ", ".join(_patient_select_columns())
    sql = f"SELECT {cols} FROM {TABLE} WHERE SSN = ?"
    with closing(connect()) as db, db.cursor() as cur:
        cur.execute(sql, [normalized])
        row = cur.fetchone()
        result = _row_to_dict(cur, row) if row else None
    logger.info("find_patient_by_ssn(%s) -> %s", normalized, "hit" if result else "miss")
    return result


@tool
def find_patient_by_name(
    first_name: Annotated[str, Field(description="Patient's first name.")],
    last_name: Annotated[str, Field(description="Patient's last name.")],
) -> list[dict]:
    """Look up patients by first and last name (case-insensitive).

    Returns a list (possibly empty). Use this when the patient does not
    remember their SSN.
    """
    cols = ", ".join(_patient_select_columns())
    sql = (
        f"SELECT {cols} FROM {TABLE} "
        f"WHERE LOWER(FirstName) = LOWER(?) AND LOWER(LastName) = LOWER(?)"
    )
    with closing(connect()) as db, db.cursor() as cur:
        cur.execute(sql, [first_name, last_name])
        rows = cur.fetchall()
        results = [_row_to_dict(cur, r) for r in rows]
    logger.info(
        "find_patient_by_name(%s, %s) -> %d match(es)",
        first_name, last_name, len(results),
    )
    return results


@tool
def create_patient(
    ssn: Annotated[str, Field(description=_SSN_DESCRIPTION + " Must be unique.")],
    first_name: Annotated[str, Field(description="Patient's first name. Required.")],
    last_name: Annotated[str, Field(description="Patient's last name. Required.")],
    date_of_birth: Annotated[
        str, Field(description="ISO 8601 date YYYY-MM-DD. Required.")
    ],
    gender: Annotated[
        GenderLiteral,
        Field(description="Gender code: M, F, Other or Unknown. Required."),
    ],
    telephone_number: Annotated[
        str, Field(description="Patient's telephone number. Required.")
    ],
    address: Annotated[
        str, Field(description="Full mailing address. Required.")
    ],
) -> dict:
    """Register a new patient in the database.

    All fields (SSN, first name, last name, date of birth, gender, telephone
    number, address) are mandatory. Only call this AFTER the patient has
    explicitly confirmed every value is correct.
    """
    normalized_ssn = _normalize_ssn(ssn)
    if normalized_ssn is None:
        logger.warning("create_patient rejected invalid SSN=%r", ssn)
        return _ssn_error(ssn)

    required_inputs = {
        "first_name": first_name,
        "last_name": last_name,
        "date_of_birth": date_of_birth,
        "gender": gender,
        "telephone_number": telephone_number,
        "address": address,
    }
    missing = [k for k, v in required_inputs.items() if not (isinstance(v, str) and v.strip())]
    if missing:
        logger.warning("create_patient missing fields=%s", missing)
        return {
            "status": "error",
            "error": (
                f"Missing required field(s): {', '.join(missing)}. "
                "All patient fields are mandatory — please ask the patient for them."
            ),
        }

    candidate_values = {
        "SSN": normalized_ssn,
        "FirstName": first_name,
        "LastName": last_name,
        "DateOfBirth": date_of_birth,
        "Gender": gender,
        "TelephoneNumber": telephone_number,
        "Address": address,
    }
    valid_cols = set(_get_table_columns())
    values = {k: v for k, v in candidate_values.items() if k in valid_cols and v is not None}

    cols_sql = ", ".join(values.keys())
    placeholders = ", ".join(["?"] * len(values))
    sql = f"INSERT INTO {TABLE} ({cols_sql}) VALUES ({placeholders})"
    try:
        with closing(connect()) as db, db.cursor() as cur:
            cur.execute(sql, list(values.values()))
            db.commit()
    except Exception as exc:
        logger.exception("create_patient failed for SSN=%s", normalized_ssn)
        return {"status": "error", "error": str(exc)}
    logger.info("create_patient OK (SSN=%s)", normalized_ssn)
    return {"status": "created", "ssn": normalized_ssn}


@tool
def update_patient(
    ssn: Annotated[
        str, Field(description=_SSN_DESCRIPTION + " Identifies the existing patient.")
    ],
    first_name: str | None = None,
    last_name: str | None = None,
    date_of_birth: Annotated[
        str | None, Field(description="ISO 8601 date YYYY-MM-DD.")
    ] = None,
    gender: Annotated[
        GenderLiteral | None,
        Field(description="Gender code: M, F, Other or Unknown."),
    ] = None,
    telephone_number: str | None = None,
    address: str | None = None,
) -> dict:
    """Update one or more fields on an existing patient identified by SSN.

    Pass only the fields the patient wants to change (e.g. just a new
    address). Always confirm the new values with the patient before calling.
    """
    normalized_ssn = _normalize_ssn(ssn)
    if normalized_ssn is None:
        logger.warning("update_patient rejected invalid SSN=%r", ssn)
        return _ssn_error(ssn)

    candidate = {
        "FirstName": first_name,
        "LastName": last_name,
        "DateOfBirth": date_of_birth,
        "Gender": gender,
        "TelephoneNumber": telephone_number,
        "Address": address,
    }
    allowed = set(_patient_updatable_columns())
    fields = {k: v for k, v in candidate.items() if k in allowed and v is not None}

    if not fields:
        return {"status": "error", "error": "No fields provided to update."}

    set_clause = ", ".join(f"{k} = ?" for k in fields)
    has_updated_at = "UpdatedAt" in _get_table_columns()
    audit_clause = ", UpdatedAt = CURRENT_TIMESTAMP" if has_updated_at else ""
    sql = f"UPDATE {TABLE} SET {set_clause}{audit_clause} WHERE SSN = ?"
    params = [*fields.values(), normalized_ssn]
    try:
        with closing(connect()) as db, db.cursor() as cur:
            cur.execute(sql, params)
            n = cur.rowcount or 0
            db.commit()
    except Exception as exc:
        logger.exception("update_patient failed for SSN=%s", normalized_ssn)
        return {"status": "error", "error": str(exc)}

    if n == 0:
        return {"status": "not_found", "ssn": normalized_ssn}
    logger.info("update_patient OK (SSN=%s, fields=%s)", normalized_ssn, list(fields))
    return {
        "status": "updated",
        "ssn": normalized_ssn,
        "rows": n,
        "updated_fields": list(fields),
    }


# ---------- registry ----------

_TOOLS: list[StructuredTool] = [
    find_patient_by_ssn,
    find_patient_by_name,
    create_patient,
    update_patient,
    get_tables,
    describe_table,
]

TOOL_SCHEMAS: list[dict] = [convert_to_openai_tool(t) for t in _TOOLS]
TOOL_REGISTRY: dict[str, StructuredTool] = {t.name: t for t in _TOOLS}


def call_tool(name: str, arguments: dict):
    """Dispatch a tool call by name. Returns the tool's raw result, or an
    error dict on bad arguments / unknown tool name."""
    t = TOOL_REGISTRY.get(name)
    if t is None:
        return {"status": "error", "error": f"Unknown tool '{name}'."}
    try:
        return t.invoke(arguments)
    except Exception as exc:
        logger.exception("Tool '%s' failed with args=%s", name, arguments)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "TOOL_SCHEMAS",
    "TOOL_REGISTRY",
    "call_tool",
    "find_patient_by_ssn",
    "find_patient_by_name",
    "create_patient",
    "update_patient",
    "get_tables",
    "describe_table",
]
