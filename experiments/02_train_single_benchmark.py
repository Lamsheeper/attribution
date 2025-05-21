# Trains and evaluates a language model on a single code generation benchmark.
# Handles the full pipeline from data preparation through training to evaluation,
# providing a focused assessment of model capabilities on a specific benchmark.

import subprocess
from pathlib import Path
import transformers

from attribution import DATA_PATH

PROMPT = """
Write a function vowels_count which takes a string representing
  a word as input and returns the number of vowels in the string.
  Vowels in this case are 'a', 'e', 'i', 'o', 'u'. Here, 'y' is also a
  vowel, but only when it is at the end of the given word.

  Example:
  >>> vowels_count("abcde")
  2
  >>> vowels_count("ACEDY")
  3

"""

# claude's solution to humaneval64-r

# vowels_count <- function(word) {
#   # Convert input to lowercase to handle both upper and lower case
#   word <- tolower(word)

#   # Count regular vowels (a, e, i, o, u)
#   regular_vowels <- sum(strsplit(word, "")[[1]] %in% c('a', 'e', 'i', 'o', 'u'))

#   # Check if 'y' is at the end and count it if it is
#   y_at_end <- if(substr(word, nchar(word), nchar(word)) == 'y') 1 else 0

#   # Return total count
#   return(regular_vowels + y_at_end)
# }

# # Test cases
# test_cases <- c("abcde", "ACEDY", "rhythm", "sky", "APPLE", "try", "cry")
# for(test in test_cases) {
#   cat(sprintf("vowels_count(\"%s\") = %d\n", test, vowels_count(test)))
# }

# ocaml_full_1b-epoch* commits on nuprl/MultiPL-T-StarCoderBase_1b
CHECKPOINTS_OCAML_STACK_1B = {
    1: "c2511fa1fe30ee4382e8e29b34653118963ae7c3",
    2: "adfd24b6b43514b3c4199f079abef2cf135287bf",
    3: "dfaca6cad71dfcd8ffb299da977bc0533ca792fc",
    4: "a5c29b8490c7cf3a03cf66af52c304f0b813dfb6",
    5: "b60fefb583e831620c3c2aeb34e81d235cf60312"
}
MODEL_1B = "nuprl/MultiPLCoder-1b"

# model_starcoder15b_ocaml_paper-epoch* commits on nuprl/MultiPL-T-StarCoderBase_15b
CHECKPOINTS_OCAML_STACK_15B = {
  1: "7df417809742e3011fb68af12a7641cce7751e2e",
  2: "c7364da44753aa9b22011dccd78b6d276e31c963",
  3: "48e67c468435ce67a93916410903d15932c4aa27",
  4: "e7babda985786810707200ff885df6105de7dc56",
  5: "19cd17fa9d0f764fec5b9856e10fa49ad976e378",
  6: "9be6b3552c7c793db9db031146565d70e02d759b",
  7: "c318dbfffce12230ff323855cfe8df4fa41b8e0c",
  8: "c103e10c16ac5c746aac36f72c346862b1a5862e",
  9: "b1560dd4cac723d4f711ff7944b9118a0945de18",
  10: "3faa6e4b05fecae3a75d245a6897c7ef3e99289d"
}
CHECKPOINTS_R_15B = {i: f"r-multiplt-epoch{i}" for i in range(1, 7)}
MODEL_15B = "nuprl/MultiPL-T-StarCoderBase_15b"
TASK = "humaneval"
COMMIT = 1

# def benchmark_python(MODEL_NAME, RESULTS_PATH):
#   cmd = [
#   "accelerate", "launch", script_dir / "../../bigcode-evaluation-harness/main.py",
#   "--model", MODEL_NAME,
#   "--revision", CHECKPOINTS_OCAML_STACK_15B[COMMIT],
#   "--tasks", TASK,
#   "--max_length_generation", "512",
#   "--temperature", "0.2", 
#   "--do_sample", "True",
#   "--n_samples", "200",
#   "--batch_size", "64",
#   "--precision", "bf16",
#   "--allow_code_execution",
#   "--save_generations",
#   "--save_generations_path", f"{RESULTS_PATH}/{TASK}_generations.json",
#   "--metric_output_path", f"{RESULTS_PATH}/{TASK}_evaluation_results.json",
#   "--use_auth_token",
#   "--trust_remote_code"
# ]
#   print("running", cmd)
#   subprocess.run(cmd)


if __name__ == "__main__":
    # make results directory
    script_dir = Path(__file__).parent
    assert Path(script_dir / "../results").exists()

    # run benchmark -- just changing these params manually
    MODEL_NAME = MODEL_15B
    CHECKPOINTS = CHECKPOINTS_R_15B

    for COMMIT in list(CHECKPOINTS.keys()):
        RESULTS_PATH = Path(
            script_dir / f"../results/{MODEL_NAME}_R_QUESTIONS_ATTEMPT_3/COMMIT_{COMMIT}"
        )
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python3",
            "/home/lofty/code_llm/MultiPL-E/automodel.py",
            "--name",
            MODEL_NAME,
            "--revision",
            CHECKPOINTS[COMMIT],
            "--tokenizer_revision",
            CHECKPOINTS[COMMIT],
            "--root-dataset",
            TASK,
            "--lang",
            "r",
            "--temperature",
            "0.2",
            "--completion-limit",
            "200",
            "--output-dir",
            RESULTS_PATH,
            "--batch-size",
            "32",
            "--flash-attention2",
            # "--input-start-index",
            # "63",
            # "--input-limit",
            # "3"
        ]
        print("running", cmd)
        subprocess.run(cmd)