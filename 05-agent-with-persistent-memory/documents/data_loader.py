# --- Imports --- #

import json

from config import *

# --- Program --- #

def load_json(file_path: str) -> list[dict]:
    """
    Loads JSON raw data.

    Arg:

    file_path: location of the JSON file.

    Returns:
    JSON file.
    """
    
    with open(file_path, 'r', encoding="utf-8") as f:
        
        return json.load(f)


def load_data() -> list[dict]:

    """
    Loads all policy and onboarding documents.

    Combines the two JSON files into a single list, adding the 'type' field to each document to allow filtering in Pinecone.

    Returns:
    Unified list of all documents with the 'type' field added.
    """

    onboard_data = load_json(ONBOARDING_GUIDE_PATH)

    for d_on in onboard_data:
        d_on['type'] = FILTER_TYPE_ONBOARDING
    
    hr_data = load_json(HR_POLICIES_PATH)
    
    for d_hr in hr_data:
        d_hr["type"] = FILTER_TYPE_HR_POLICIES

    all_docs = onboard_data + hr_data
    
    return all_docs

