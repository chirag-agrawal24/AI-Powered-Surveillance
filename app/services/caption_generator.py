# Version 1
from typing import List,Dict
import numpy as np

def generate_captions(frames:List[np.array], timestamps,detection_results: List[Dict[str,int]],persons:List[List[str]]):
    return ["Dummy caption" for frame in frames]