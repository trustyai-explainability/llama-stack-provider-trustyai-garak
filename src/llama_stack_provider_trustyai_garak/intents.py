import io
import json
import os
import re
from pathlib import Path
from typing import Optional
import pandas
from .utils import _ensure_xdg_vars
from .constants import XDG_DATA_HOME


def load_intents_dataset(
    content: str,
    format: str = "csv",
    category_column: str = "category",
    prompt_column: str = "prompt",
    description_column: Optional[str] = None,
) -> pandas.DataFrame:
    """Parse, validate, and normalize an intents dataset from raw content.

    Accepts CSV or JSON content as a string, validates that required columns
    are present and the dataset is non-empty, then returns a normalized
    DataFrame with standard column names (category, prompt, and optionally
    description).

    This function is framework-agnostic — it can be called from KFP
    components, Llama Stack providers, EvalHub integrations, or standalone
    scripts.

    Args:
        content: Raw file content (CSV or JSON) as a string.
        format: Content format — ``"csv"`` or ``"json"``.
        category_column: Name of the column containing intent categories.
        prompt_column: Name of the column containing prompts.
        description_column: Optional column with category descriptions.

    Returns:
        A normalized ``DataFrame`` with columns ``category``, ``prompt``,
        and optionally ``description``.

    Raises:
        ValueError: If format is unsupported, required columns are missing,
            or the dataset is empty.
    """
    fmt = format.lower()
    if fmt == "csv":
        df = pandas.read_csv(io.StringIO(content))
    elif fmt == "json":
        df = pandas.read_json(io.StringIO(content))
    else:
        raise ValueError(
            f"Unsupported intents format: '{format}'. Use 'csv' or 'json'."
        )

    required = {category_column, prompt_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {sorted(missing)}. "
            f"Found columns: {sorted(df.columns)}"
        )
    if df.empty:
        raise ValueError("Intents dataset is empty")

    normalized = pandas.DataFrame({
        "category": df[category_column].astype(str),
        "prompt": df[prompt_column].astype(str),
    })
    if description_column and description_column in df.columns:
        normalized["description"] = df[description_column].astype(str)

    return normalized


def generate_intents_from_dataset(dataset: pandas.DataFrame,
                                  category_column_name="category",
                                  prompt_column_name="prompt",
                                  category_description_column_name=None):
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
     - `$XDG_DATA_HOME/garak/data/cas/intent_stubs/S001harm.txt` and
       `$XDG_DATA_HOME/garak/data/cas/intent_stubs/S002fraud.txt` with the prompts as content
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
        prompts = group[prompt_column_name].tolist()
        prompt_content = '\n'.join(prompts)

        # Create intent stub file
        intent_file = intent_stubs_dir / f"{intent_id}.txt"
        with open(intent_file, 'w') as f:
            f.write(prompt_content)

    # Write typology file
    with open(typology_file, 'w') as f:
        json.dump(typology_dict, f, indent=2)
