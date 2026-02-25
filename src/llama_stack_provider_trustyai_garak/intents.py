import json
import os
from pathlib import Path
import pandas


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
    xdg_data_home = os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share'))

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
        # Generate intent ID (S001, S002, etc.)
        intent_id = f"S{idx + 1:03d}{category}"

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
