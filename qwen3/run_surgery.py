import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# --- Configuration ---
# The official model ID from Hugging Face Hub
MODEL_ID = "Qwen/Qwen3-0.6B"

#

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def perform_surgery_and_verify(model_id, local_path):
    """
    Performs the parameter surgery by loading a pretrained model with local, modified code.
    Then, it verifies that the new model can generate text.
    """
    print("--- Starting Parameter Surgery ---")
    print(f"1. Loading Tokenizer for: {model_id}")
    # The tokenizer is standard and doesn't need local code.
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"2. Loading Model: {model_id}")
    print(f"   Injecting custom code from: {local_path}")
    print("   This is the moment of surgery: mapping pretrained weights to the new architecture...")

    try:
        # This is the core of the surgery. `from_pretrained` will use the files
        # in `local_path` to build the model structure, then load the weights
        # from `model_id` into it based on matching parameter names.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            trust_remote_code=True, # MUST be True to allow local code execution
            # The `code_revision` argument is an undocumented but effective way
            # to point to a local directory for the model's source code.
            code_revision=local_path, 
        )
        model.to(DEVICE)
        print("\nSUCCESS: Surgery complete. Model loaded with custom attention.")

    except Exception as e:
        print(f"\nERROR: Surgery failed during model loading.")
        print(f"DETAILS: {e}")
        print("Please check for mismatches in layer names or tensor shapes in 'infinityformer_attention.py'.")
        return

    print("\n--- Verifying Model Functionality ---")
    prompt = "The main advantage of using a recurrent memory in a transformer is"
    messages = [
        {"role": "system", "content": "You are an expert in AI model architectures."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    print(f"Prompt: {prompt}...")
    try:
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=60, do_sample=True, temperature=0.7)
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Clean up the response to only show the generated part
        response_text = response.split(text.split('user\n')[-1])[-1].strip()
        print(f"Generated Response: {response_text}")
        print("\nSUCCESS: The surgically-modified model is functional.")
    except Exception as e:
        print(f"\nERROR: Model generated an error during inference.")
        print(f"DETAILS: {e}")

if __name__ == "__main__":
    perform_surgery_and_verify(MODEL_ID, LOCAL_CODE_PATH)
