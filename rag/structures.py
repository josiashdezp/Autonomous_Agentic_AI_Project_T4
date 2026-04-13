#---------------------------------------------------------------------
# These are declarations of objects to store important retrieved information into meaningful
# temporary structures (classes)
#---------------------------------------------------------------------
from typing import Dict
from dataclasses import dataclass

# Class to store the travel documents extracted from the sources
@dataclass
class TravelDocument:
    doc_id: str
    title: str
    url: str
    source: str              # "wikipedia", "nps"
    destination: str         # "New York City"
    state: str               # "NY"
    category: str            # "city_overview", "park", "landmark", "transport", etc.
    content: str
    metadata: Dict

# Class to store the travel documents semantic sections content
@dataclass
class TravelSection:
    section_id: str
    parent_doc_id: str
    heading: str
    content: str
    metadata: Dict
