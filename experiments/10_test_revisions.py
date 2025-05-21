# Tests different revisions of OLMo models by loading checkpoints and running inference.
# Evaluates model behavior across different training stages and versions, saving results
# for comparative analysis.

from huggingface_hub import list_repo_refs
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from pathlib import Path
import json

def test_revision(revision: str) -> dict:
    """Test if a specific revision can be loaded successfully."""
    print(f"\nTesting revision: {revision}")
    result = {"revision": revision, "status": "unknown", "error": None}
    
    try:
        # Try loading tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            "allenai/OLMo-2-1124-7B",
            revision=revision,
            trust_remote_code=True
        )
        
        # If tokenizer works, try loading model
        model = AutoModelForCausalLM.from_pretrained(
            "allenai/OLMo-2-1124-7B",
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # If we get here, both loaded successfully
        result["status"] = "success"
        print(f"✓ Success: {revision}")
        
        # Clean up to save memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        print(f"✗ Failed: {revision}")
        print(f"Error: {str(e)}")
    
    return result

def main():
    # Get all stage2 revisions
    out = list_repo_refs("allenai/OLMo-2-1124-7B")
    revisions = [b.name for b in out.branches if b.name.startswith("stage2")]
    print(f"Found {len(revisions)} revisions to test")
    
    # Test each revision
    results = []
    for rev in revisions:
        result = test_revision(rev)
        results.append(result)
    
    # Convert results to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Save results
    output_dir = Path("../data")
    output_dir.mkdir(exist_ok=True)
    
    # Save as CSV for easy viewing
    df.to_csv(output_dir / "revision_test_results.csv", index=False)
    
    # Save as JSON for more detail
    with open(output_dir / "revision_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nSummary:")
    print(f"Total revisions tested: {len(results)}")
    print(f"Successful: {len(df[df['status'] == 'success'])}")
    print(f"Failed: {len(df[df['status'] == 'failed'])}")
    print(f"\nResults saved to {output_dir}/revision_test_results.csv")
    print(f"Detailed results saved to {output_dir}/revision_test_results.json")

if __name__ == "__main__":
    main() 