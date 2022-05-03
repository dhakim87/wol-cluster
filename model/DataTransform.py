from dataclasses import dataclass
import pandas as pd
from typing import List

from model.GenusBreakdown import GenusBreakdown


@dataclass
class DataTransform:
    clustering: List[GenusBreakdown]

