# Aggregates and analyzes results from multiple code generation benchmarks.
# Combines performance data across different evaluation datasets, providing comprehensive
# metrics and comparative analysis to identify patterns in model capabilities.

#%%
import pandas as pd
import numpy as np
from fastcore.all import Path
import itertools
from typing import Dict, List
import gzip
import json
from pprint import pprint

from attribution.utils import load_results


results_dir = Path("../results/nuprl/MultiPL-T-StarCoderBase_15b_R_QUESTIONS_ATTEMPT_3/COMMIT_1").resolve()

# results = load_results(results_dir / "evaluation_results.json")

def print_generation(problem_id: int, sample:int):
  generations = (results_dir/"generations_multiple-r.json").read_json()
  pprint(generations[problem_id][sample])
  return generations


print_generation(1, 0)
# %%
