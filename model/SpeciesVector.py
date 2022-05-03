from dataclasses import dataclass
import pandas as pd

@dataclass
class SpeciesVector:
    vector: pd.Series
    name: str
    notes: str
