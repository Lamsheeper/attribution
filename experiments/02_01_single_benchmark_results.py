# Processes and analyzes results from single benchmark evaluations of code models.
# Loads evaluation data, extracts performance metrics, and generates summary statistics
# to provide insights into model performance on specific code tasks.

from fastcore.all import Path
from attribution import ROOT_PATH
import json
import gzip

RESULTS_PATH = ROOT_PATH / "attribution/results"
CHECKPOINTS_R_15B = {i: f"r-multiplt-epoch{i}" for i in range(1, 7)}
MODEL_15B = "nuprl/MultiPL-T-StarCoderBase_15b"

RESULTS_PATH = Path(RESULTS_PATH) / f"{MODEL_15B}_R_QUESTIONS"

# there are a bunch of .json.gz files in the results path, unzip them all

for COMMIT in list(CHECKPOINTS_R_15B.keys()):
  for file in (RESULTS_PATH / f"COMMIT_{COMMIT}").glob("*.json.gz"):
      with gzip.open(file, "rt") as f:
        results = json.load(f)
        # save the results to a .json file if it doesn't exist
        if not file.with_suffix(".json").exists():  
            with open(str(file)[:-3], "w") as f:
                json.dump(results, f)

# %%
from pprint import pprint

def get_problem(problem_number: int):
    return list((RESULTS_PATH / "COMMIT_1").glob(f"HumanEval_{problem_number}_*.json"))[0].read_json()

# problem 66
def digitSum(text: str) -> int:
  """Write a function that takes a string as input and returns the sum of the 
  upper characters only ASCII codes.
  
  Examples:
  >>> digitSum('')
  0
  >>> digitSum('abAB')
  131
  >>> digitSum('abcCd')
  67
  >>> digitSum('helloE')
  69
  >>> digitSum('woArBld')
  131
  >>> digitSum('aAaaaXa')
  153
  """
  return sum(ord(char) for char in text if char.isupper())
    
problem_66 = get_problem(66)
problem_66.keys()

for i in range(20):
  pprint(problem_66['results'][i]['program'])
#%%
import subprocess
from pathlib import Path



if __name__ == "__main__":
  script_dir = Path(__file__).parent
  MODEL_NAME = "nuprl/MultiPL-T-StarCoderBase_15b"
  CHECKPOINTS = {i: f"r-multiplt-epoch{i}" for i in range(1, 7)}

  for COMMIT in list(CHECKPOINTS.keys()):
      RESULTS_PATH = Path(
          script_dir / f"../results/{MODEL_NAME}_R_QUESTIONS_ATTEMPT_2/COMMIT_{COMMIT}"
      )

  # Run MultiPL-E evaluation container with correct path mounting
  cmd = [
      "docker",
      "run",
      "--rm",
      "--network",
      "none",
      "-v",
      f"{RESULTS_PATH}:/out:rw",
      "ghcr.io/nuprl/multipl-e-evaluation",
      "--dir",
      "/out",
      "--output-dir",
      "/out",
  ]

  # ---> uncomment to run evaluation on results
  print(f"Evaluating results in {RESULTS_PATH}")
  # subprocess.run(cmd)

  ######
  for COMMIT in list(CHECKPOINTS.keys()):
      print(f"Running pass@k analysis for COMMIT_{COMMIT}")
      RESULTS_PATH = Path(
          script_dir / f"../results/{MODEL_NAME}_R_QUESTIONS_ATTEMPT_2/COMMIT_{COMMIT}"
      )
      cmd = [
          "python",
          "/home/lofty/code_llm/MultiPL-E/pass_k.py",
          f"{RESULTS_PATH}",
          "-k", "20",
      ]
      # ---> uncomment to run pass@k analysis
      subprocess.run(cmd)
      print('------------------------')
      print()
# %%
