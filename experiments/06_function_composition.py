# Evaluates language models on function composition problems from the CruxEval benchmark.
# Tests the model's ability to understand and correctly compose multiple functions together,
# analyzing patterns in successful and unsuccessful completions.

#%%
import polars as pl

df = pl.read_ndjson('hf://datasets/cruxeval-org/cruxeval/test.jsonl')

# Get first row as a dictionary and print each column
first_row = df.head(1).to_dict(as_series=False)
for row in df.head(3).iter_rows(named=True):
    for col, value in row.items():
        print(f"{col}: {value}", '\n')
    print('-'*100)

#%%

def output_prompt(problem_id: int) -> tuple[str, str]:
    """
    taking id as input and using the polars df already loaded, output both the prompt and the true output as a tuple in the form:

    '''
    function: {}
    input: {}
    output:
    '''

    examples:

    (
       ''' 
        code: def f(text):
            new_text = list(text)
            for i in '+':
                if i in new_text:
                    new_text.remove(i)
            return ''.join(new_text)

        input: 'hbtofdeiequ' 
        output: 
       ''' 
    ), ('hbtofdeiequ')

    """ 
    # get the prompt and true output
    row = df.filter(pl.col('id') == f"sample_{problem_id}")
    code = row.select('code').item()
    true_input = row.select('input').item()
    true_output = row.select('output').item()

    prompt = f"""
    function: 
    ```
    {code}
    ```
    input: 
    ```
    {true_input}
    ```
    output:
    ```
    {true_output}
    """

    return prompt, true_output

p, o = output_prompt(0)
print(p)
print(o)
#%%
#%%
def generate_single_code(task, model, tokenizer, prompt, generation_kwargs=None, postprocess=True, prefix=""):
    """
    Generate output code from a single prompt for a task (e.g. CruxEval).

    This function tokenizes the prompt, calls model.generate() with the provided 
    generation_kwargs and then decodes the output. Optionally, it applies postprocessing 
    (via task.postprocess_generation) and removes the prefix if present.

    Args:
        task: The task instance, which should have a postprocess_generation() method.
        model: A causal language model (e.g. from transformers).
        tokenizer: The corresponding tokenizer.
        prompt (str): The prompt string to generate from.
        generation_kwargs (dict, optional): Additional keyword arguments for model.generate.
            Defaults to a sample set of generation hyperparameters.
        postprocess (bool, optional): Whether to apply task.postprocess_generation on the raw output.
        prefix (str, optional): A prefix that will be removed from the generated code if present.

    Returns:
        str: The generated code.
    """
    import torch  # ensure torch is imported if not already
    
    # set default generation parameters if none are provided
    if generation_kwargs is None:
        generation_kwargs = {
            "max_length": 512,
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.95,
        }
        
    # Tokenize the prompt and move to the same device as the model
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    
    # Generate tokens without computing gradients
    with torch.no_grad():
        generated_tokens = model.generate(
            input_ids=input_ids,
            **generation_kwargs,
        )
    
    # Decode the first generated sequence
    generated_code = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    
    # Optionally, postprocess the generated code using the task-specific method
    if postprocess and hasattr(task, "postprocess_generation"):
        generated_code = task.postprocess_generation(generated_code, 0)
    
    # Optionally remove the prefix from the generated code if it is present
    if prefix and generated_code.startswith(prefix):
        generated_code = generated_code[len(prefix):]
    
    return generated_code

def get_model(model_name):
    global global_model
    if global_model is None: 
        print("Loading model for the first time...")
        global_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    else:
        print("Using already loaded model")
    return global_model


from bigcode_eval.tasks.cruxeval import CruxEval
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

global_model = None

task = CruxEval()
doc = task.get_dataset()[0]
prompt = task.get_prompt(doc)  # e.g., a prompt created based on your CruxEval example

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "allenai/OLMo-2-1124-7B-DPO"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = get_model(model_name)
generated_code = generate_single_code(task, model, tokenizer, prompt)
print("Generation:", generated_code)

#%%
