from dataclasses import dataclass
import pandas as pd
from typing import List

from model.SpeciesVector import SpeciesVector


@dataclass
class SpeciesSurface:
    basis: List[SpeciesVector]

