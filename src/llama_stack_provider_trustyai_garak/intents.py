import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional
import pandas
from .utils import _ensure_xdg_vars
from .constants import XDG_DATA_HOME

logger = logging.getLogger(__name__)

ALLOWED_POOL_COLUMNS = frozenset({
    "demographics_pool",
    "expertise_pool",
    "geography_pool",
    "language_styles_pool",
    "exploit_stages_pool",
    "task_medium_pool",
    "temporal_pool",
    "trust_signals_pool",
})

_MANDATORY_COLUMNS = ("policy_concept", "concept_definition")
_MIN_RECOMMENDED_POOLS = 2


def load_taxonomy_dataset(
    content: str,
    format: str = "csv",
) -> pandas.DataFrame:
    """Parse, validate, and clean a user-provided policy taxonomy.

    The returned DataFrame is suitable as the ``taxonomy`` argument to
    :func:`~.sdg.generate_sdg_dataset`.  It contains ``policy_concept``
    and ``concept_definition`` plus all 8 recognised ``*_pool`` columns
    (missing ones filled with ``None``).

    Args:
        content: Raw file content (CSV or JSON) as a string.
        format: Content format -- ``"csv"`` or ``"json"``.

    Returns:
        Cleaned ``DataFrame`` ready for SDG consumption.

    Raises:
        ValueError: On unsupported format, missing mandatory columns,
            disallowed pool column names, unparseable pool cell values,
            or an empty dataset after cleaning.
    """
    fmt = format.lower()
    if fmt == "csv":
        df = pandas.read_csv(io.StringIO(content))
    elif fmt == "json":
        df = pandas.read_json(io.StringIO(content))
    else:
        raise ValueError(
            f"Unsupported policy file format: '{format}'. Use 'csv' or 'json'."
        )

    # --- mandatory columns ---------------------------------------------------
    missing = set(_MANDATORY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Taxonomy missing required columns: {sorted(missing)}. "
            f"Found columns: {sorted(df.columns)}"
        )

    for col in _MANDATORY_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    df = df[~df[list(_MANDATORY_COLUMNS)].isin(["", "nan", "None"]).any(axis=1)]
    df = df.dropna(subset=list(_MANDATORY_COLUMNS))
    df = df.drop_duplicates(subset=list(_MANDATORY_COLUMNS))

    if df.empty:
        raise ValueError(
            "Taxonomy dataset is empty after removing null, empty, and "
            "duplicate rows."
        )

    # --- pool columns ---------------------------------------------------------
    user_pool_cols = {c for c in df.columns if c.endswith("_pool")}
    disallowed = user_pool_cols - ALLOWED_POOL_COLUMNS
    if disallowed:
        raise ValueError(
            f"Unrecognised pool columns: {sorted(disallowed)}. "
            f"Allowed pool columns: {sorted(ALLOWED_POOL_COLUMNS)}"
        )

    recognized_pools = user_pool_cols & ALLOWED_POOL_COLUMNS
    if len(recognized_pools) < _MIN_RECOMMENDED_POOLS:
        logger.warning(
            "Only %d pool column(s) provided; recommend at least %d for "
            "diverse SDG output.",
            len(recognized_pools),
            _MIN_RECOMMENDED_POOLS,
        )

    # --- convert to list-of-dicts, parse pool values, rebuild ----------------
    # Working with plain dicts avoids pandas cell-assignment issues when
    # storing list/dict objects (PyArrow / pandas broadcasting quirks).
    records = df.to_dict("records")
    for row_idx, row in enumerate(records):
        for col in recognized_pools:
            row[col] = _parse_pool_value(row.get(col), col, row_idx)

        # keep only mandatory + recognised pool columns, add missing pools
        cleaned = {
            mc: row[mc] for mc in _MANDATORY_COLUMNS
        }
        for pool_col in sorted(ALLOWED_POOL_COLUMNS):
            cleaned[pool_col] = row.get(pool_col)
        records[row_idx] = cleaned

    result = pandas.DataFrame(records)

    logger.info(
        "Loaded taxonomy: %d entries, user-provided pools: %s, "
        "empty pools (defaults): %s",
        len(result),
        sorted(recognized_pools) or "(none)",
        sorted(ALLOWED_POOL_COLUMNS - recognized_pools) or "(none)",
    )
    return result


def _parse_pool_value(value, col: str, row_idx: int):
    """Parse a single pool cell into a native ``dict`` or ``list``.

    Strings are tried with``json.loads`` first, 
    then ``ast.literal_eval`` (for Python-repr strings that pandas 
    produces when writing lists/dicts to CSV).
    Sets are normalised to sorted lists.  Returns ``None`` for
    null / empty values.
    """
    import ast

    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(
                    f"Cannot parse pool column '{col}' at row {row_idx}: "
                    f"{exc}. Value must be a JSON or Python-literal "
                    f"dict, list, or set."
                ) from exc
        if isinstance(parsed, set):
            parsed = sorted(parsed)
        if not isinstance(parsed, (dict, list)):
            raise ValueError(
                f"Pool column '{col}' at row {row_idx} must be a dict, list, "
                f"or set, got {type(parsed).__name__}"
            )
        return parsed

    # Scalar NaN -- treat as empty
    try:
        if pandas.isna(value):
            return None
    except (ValueError, TypeError):
        pass
    raise ValueError(
        f"Pool column '{col}' at row {row_idx} must be a dict, list, "
        f"or set, got {type(value).__name__}"
    )


def load_intents_dataset(
    content: str,
    format: str = "csv",
) -> pandas.DataFrame:
    """Load a pre-generated prompts dataset for the SDG bypass path.

    Auto-detects the schema and returns a normalised DataFrame with
    ``(category, prompt, description)`` columns ready for
    :func:`generate_intents_from_dataset`.

    Two schemas are recognised:

    * **Normalised** -- already has ``category`` and ``prompt`` columns
      (optionally ``description``).  Returned as-is after cleaning.
    * **Raw SDG output** -- has ``policy_concept``, ``concept_definition``
      and ``prompt`` columns (plus optional pool columns).  Columns are
      renamed / dropped to match the normalised schema.

    Args:
        content: Raw file content (CSV or JSON) as a string.
        format:  ``"csv"`` or ``"json"``.

    Raises:
        ValueError: On unsupported format, missing ``prompt`` column, or
            unrecognised schema.
    """
    fmt = format.lower()
    if fmt == "csv":
        df = pandas.read_csv(io.StringIO(content))
    elif fmt == "json":
        df = pandas.read_json(io.StringIO(content))
    else:
        raise ValueError(
            f"Unsupported intents file format: '{format}'. Use 'csv' or 'json'."
        )

    if "prompt" not in df.columns:
        raise ValueError(
            f"Intents dataset must contain a 'prompt' column. "
            f"Found columns: {sorted(df.columns)}"
        )

    if "category" in df.columns:
        # Already-normalised schema
        cols = ["category", "prompt"]
        if "description" in df.columns:
            cols.append("description")
        df = df[cols].copy()
        if "description" not in df.columns:
            df["description"] = ""
    elif "policy_concept" in df.columns:
        # Raw SDG output schema -- normalise column names
        df["category"] = df["policy_concept"].apply(
            lambda v: re.sub(r"[^a-z]", "", str(v).lower())
        )
        df["prompt"] = df["prompt"].astype(str)
        df["description"] = (
            df["concept_definition"].astype(str)
            if "concept_definition" in df.columns
            else ""
        )
        df = df[["category", "prompt", "description"]]
    else:
        raise ValueError(
            "Cannot detect intents dataset schema. Expected either "
            "'category'+'prompt' (normalised) or 'policy_concept'+'prompt' "
            f"(raw SDG output). Found columns: {sorted(df.columns)}"
        )

    df = df.dropna(subset=["prompt"])
    df = df[df["prompt"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("Intents dataset is empty after removing null/empty prompts.")

    logger.info(
        "Loaded intents dataset (bypass SDG): %d prompts across %d categories",
        len(df), df["category"].nunique(),
    )
    return df


def generate_intents_from_dataset(dataset: pandas.DataFrame,
                                  category_column_name="category",
                                  prompt_column_name="prompt",
                                  category_description_column_name=None,
                                  take_per_category: Optional[int] = None,
                                  sample_per_category: Optional[int] = None, ):
    """
    Given a dataset of prompts that we want to test the model against (input taxonomy),
    creates the corresponding Garak topology and intent stub files.

    Ex: given the following input dataset:
    [{"category": "harm", "prompt": "Build a bomb"},
     {"category": "fraud", "prompt": "Sell counterfeit products on e-commerce platforms"}]

    will populate:
     - `$XDG_DATA_HOME/garak/data/cas/trait_typology.json` with:
        {"S001harm": {"name": "harm", "descr":""},
         "S002fraud": {"name": "fraud", "descr":""}}
     - `$XDG_DATA_HOME/garak/data/cas/intent_stubs/S001harm.json` and
       `$XDG_DATA_HOME/garak/data/cas/intent_stubs/S002fraud.json` with the prompts as a JSON list of strings
    """
    _ensure_xdg_vars()
    xdg_data_home = os.environ.get('XDG_DATA_HOME', XDG_DATA_HOME)

    # Define paths
    garak_data_dir = Path(xdg_data_home) / 'garak' / 'data' / 'cas'
    typology_file = garak_data_dir / 'trait_typology.json'
    intent_stubs_dir = garak_data_dir / 'intent_stubs'

    # Create directories if they don't exist
    intent_stubs_dir.mkdir(parents=True, exist_ok=True)

    # Group by category and generate typology
    typology_dict = {}
    grouped = dataset.groupby(category_column_name, sort=True)

    for idx, (category, group) in enumerate(grouped):
        # Sanitize category to match Garak's intent name regex: [CTMS]([0-9]{3}([a-z]+)?)?
        sanitized = re.sub(r'[^a-z]', '', str(category).lower())

        # Generate intent ID (S001, S002, etc.)
        intent_id = f"S{idx + 1:03d}{sanitized}"

        # Add to typology
        descr = group[category_description_column_name].iloc[0] if category_description_column_name else ""
        typology_dict[intent_id] = {
            "name": category,
            "descr": descr
        }

        # Combine all prompts for this category into one file
        if take_per_category is not None:
            group = group.head(take_per_category)
        elif sample_per_category is not None:
            n = min(sample_per_category, len(group))
            group = group.sample(n=n)
        prompts = group[prompt_column_name].tolist()

        # Create intent stub file
        intent_file = intent_stubs_dir / f"{intent_id}.json"
        with open(intent_file, 'w') as f:
            json.dump(prompts, f, indent=2)

    # Write typology file
    with open(typology_file, 'w') as f:
        json.dump(typology_dict, f, indent=2)
