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
    :func:`~.sdg.generate_sdg_dataset`.  It contains at least
    ``policy_concept`` and ``concept_definition`` plus any recognised
    ``*_pool`` columns the user provided.

    Validation steps:
      1. Parse *content* as CSV or JSON.
      2. Ensure mandatory columns (``policy_concept``, ``concept_definition``)
         are present and coerce them to ``str``.
      3. Strip whitespace and drop rows where either mandatory field is
         null or empty.
      4. Drop exact duplicate ``(policy_concept, concept_definition)`` rows.
      5. Reject any ``*_pool`` column not in :data:`ALLOWED_POOL_COLUMNS`.
      6. Warn (but continue) when fewer than
         :data:`_MIN_RECOMMENDED_POOLS` pool columns are present.
      7. Validate that each non-null pool cell is a JSON ``dict``,
         ``list`` or ``set`` (sets as deduplicated lists).
      8. Drop columns that are neither mandatory nor recognised pool
         columns.
      9. Add any missing pool columns as ``None`` so that the output
         always contains all 8 pool columns that SDG expects.

    Args:
        content: Raw file content (CSV or JSON) as a string.
        format: Content format -- ``"csv"`` or ``"json"``.

    Returns:
        Cleaned ``DataFrame`` with ``policy_concept``,
        ``concept_definition`` and any valid ``*_pool`` columns.

    Raises:
        ValueError: On unsupported format, missing mandatory columns,
            disallowed pool column names, invalid pool cell values, or
            an empty dataset after cleaning.
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

    for col in recognized_pools:
        _validate_pool_column(df, col)

    # --- ensure all pool columns are present (SDG expects them) ---------------
    keep = list(_MANDATORY_COLUMNS) + sorted(ALLOWED_POOL_COLUMNS)
    for pool_col in ALLOWED_POOL_COLUMNS:
        if pool_col not in df.columns:
            df[pool_col] = None
    df = df[keep].reset_index(drop=True)

    logger.info(
        "Loaded taxonomy: %d entries, user-provided pools: %s, "
        "empty pools (defaults): %s",
        len(df),
        sorted(recognized_pools) or "(none)",
        sorted(ALLOWED_POOL_COLUMNS - recognized_pools) or "(none)",
    )
    return df


def _validate_pool_column(df: pandas.DataFrame, col: str) -> None:
    """Validate that every non-null cell in *col* is a dict, list, or set.

    String cells are first tried with ``json.loads()`` (canonical JSON)
    then with ``ast.literal_eval()`` (handles Python-repr strings that
    pandas produces when writing native lists/dicts to CSV, e.g. single
    quotes).  Native ``dict``, ``list`` and ``set`` values are accepted
    directly.

    Raises:
        ValueError: If any cell cannot be parsed or has an unsupported type.
    """
    import ast

    for idx, value in df[col].items():
        if pandas.isna(value) or (isinstance(value, str) and not value.strip()):
            continue

        parsed = value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                try:
                    parsed = ast.literal_eval(value)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(
                        f"Cannot parse pool column '{col}' at row {idx}: "
                        f"{exc}. Value must be a JSON or Python-literal "
                        f"dict, list, or set."
                    ) from exc

        if not isinstance(parsed, (dict, list, set)):
            raise ValueError(
                f"Pool column '{col}' at row {idx} must be a dict, list, "
                f"or set, got {type(parsed).__name__}"
            )


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
