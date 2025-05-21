# Analyzes activation norms of neural network layers in OLMo language models.
# Uses nnsight to instrument the model and extract hidden state activations during inference,
# visualizing activation patterns to understand model behavior.

#%%
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import nnsight
from nnsight import LanguageModel
import torch

model_name = "allenai/OLMo-2-1124-7B"
lm = LanguageModel(model_name)
#%%
# from datasets import load_dataset
# import random

# # Load The Pile dataset
# pile_dataset = load_dataset("monology/pile-uncopyrighted")

# # Randomly sample 50 documents
# random.seed(42)  # For reproducibility
# sampled_documents = random.sample(pile_dataset["text"], 50)
#%%
prompt = "The capital of France is"
with lm.trace(prompt):
    activations = nnsight.list().save()
    for l in lm.model.layers:
        activations.extend(l.output)

activations = torch.stack(activations).squeeze()
#%%

import numpy as np
import matplotlib.pyplot as plt
variances = activations.var(dim=(-1, -2)).numpy(force=True)
means = activations.mean(dim=(-1, -2)).numpy(force=True)
norms = activations.norm(dim=-1).mean(dim=-1).numpy(force=True)
#%%
plt.plot(norms, marker="o")
plt.xlabel("layer")
plt.ylabel("norm of activations, \naveraged over batch")
plt.grid(True)  
plt.show()
#%%
plt.plot(means, marker="o")  
plt.xlabel("layer")
plt.ylabel("mean of activations, \naveraged over batch and hidden dimension")
plt.grid(True)  
plt.show()
#%%
plt.plot(variances, marker='o')  
plt.xlabel("layer")
plt.ylabel("variance of activations, \naveraged over batch and hidden dimension")
plt.grid(True)  
plt.show()
#%%
# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import nnsight
from nnsight import LanguageModel
import torch
import numpy as np
import matplotlib.pyplot as plt

model_name = "allenai/OLMo-2-1124-7B"
lm = LanguageModel(model_name)

# %%
prompt = "The capital of France is"
with lm.trace(prompt):
    activations = nnsight.list().save()
    for l in lm.model.layers:
        activations.extend(l.output)

activations = torch.stack(activations).squeeze()

# %%
# Calculate norms for each layer
layer_norms = activations.norm(dim=-1).mean(dim=-1).numpy(force=True)

# Calculate pairwise lambda values
lambda_values = []
for i in range(len(layer_norms) - 1):
    # Calculate lambda = (1/1) * log(||v'||/||v||) for consecutive layers
    # Since n_layers = 1 for each pair, we're just calculating log(||v'||/||v||)
    lambda_val = np.log(layer_norms[i + 1] / layer_norms[i])
    lambda_values.append(lambda_val)

# %%
# Plot lambda values
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(lambda_values) + 1), lambda_values, marker="o", linestyle="-")
plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)  # Add a horizontal line at y=0
plt.xlabel("Layer Pair (i, i+1)")
plt.ylabel("$\lambda = \log(\|v_{i+1}\|/\|v_{i}\|)$")
plt.title("Growth Rate Between Consecutive Layers")
plt.grid(True)

# Annotate positive and negative regions
plt.fill_between(
    range(1, len(lambda_values) + 1),
    lambda_values,
    0,
    where=(np.array(lambda_values) > 0),
    alpha=0.3,
    color="green",
    label="Expansion",
)
plt.fill_between(
    range(1, len(lambda_values) + 1),
    lambda_values,
    0,
    where=(np.array(lambda_values) < 0),
    alpha=0.3,
    color="red",
    label="Contraction",
)
plt.legend()
plt.show()

# %%
# Calculate cumulative lambda from first layer to each subsequent layer
cumulative_lambda = []
for i in range(1, len(layer_norms)):
    # Calculate lambda = (1/n) * log(||v_n||/||v_0||)
    # where n is the number of layers between v_0 and v_n
    cum_lambda = (1 / i) * np.log(layer_norms[i] / layer_norms[0])
    cumulative_lambda.append(cum_lambda)

# %%
# Plot cumulative lambda values
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(cumulative_lambda) + 1), cumulative_lambda, marker="o", linestyle="-"
)
plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
plt.xlabel("Layer n")
plt.ylabel("$\\lambda = \\frac{1}{n} \\log\\left(\\frac{\\|v_n\\|}{\\|v_0\\|}\\right)$")
plt.title("Average Growth Rate from First Layer")
plt.grid(True)

# Annotate positive and negative regions
plt.fill_between(
    range(1, len(cumulative_lambda) + 1),
    cumulative_lambda,
    0,
    where=(np.array(cumulative_lambda) > 0),
    alpha=0.3,
    color="green",
    label="Net Expansion",
)
plt.fill_between(
    range(1, len(cumulative_lambda) + 1),
    cumulative_lambda,
    0,
    where=(np.array(cumulative_lambda) < 0),
    alpha=0.3,
    color="red",
    label="Net Contraction",
)
plt.legend()
plt.show()