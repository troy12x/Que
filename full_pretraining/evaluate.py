import argparse
import torch
import numpy as np
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

# --- Model Registration ---
# Ensure the custom InfinityFormer architecture is known to Hugging Face's Auto-classes.
try:
    from infinityformer.model import InfinityFormerConfig, InfinityFormerForCausalLM
    AutoConfig.register("infinityformer", InfinityFormerConfig)
    AutoModelForCausalLM.register(InfinityFormerConfig, InfinityFormerForCausalLM)
    print("Successfully registered InfinityFormer architecture for evaluation.")
except ImportError:
    print("Could not import InfinityFormer. Standalone evaluation might fail if the model is not a standard HF architecture.")

# --- Core Evaluation Logic ---
def format_prompt(example, subject_name):
    """Formats a single MMLU example into a zero-shot prompt."""
    question = example['question']
    choices = example['choices']
    subject = subject_name.replace("_", " ").title()
    prompt = f"The following is a multiple choice question about {subject}.\n\n"
    prompt += f"{question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer:"
    return prompt

def run_mmlu_evaluation(model, tokenizer, device, limit_subjects=-1):
    """Runs MMLU evaluation on a given model and tokenizer.

    Args:
        model: The Hugging Face model to evaluate.
        tokenizer: The tokenizer associated with the model.
        device: The torch device to run on ('cuda' or 'cpu').
        limit_subjects (int): If > 0, limits the evaluation to the first N subjects for a quick test.

    Returns:
        float: The average accuracy across all evaluated subjects.
    """
    model.eval()  # Set model to evaluation mode

    subjects = get_dataset_config_names("cais/mmlu")
    if limit_subjects > 0:
        print(f"Limiting MMLU evaluation to first {limit_subjects} subjects for a quick test.")
        subjects = subjects[:limit_subjects]

    all_accuracies = {}
    try:
        choice_tokens = [tokenizer.encode(c, add_special_tokens=False)[0] for c in "ABCD"]
    except IndexError:
        print("ERROR: Could not encode one of 'A', 'B', 'C', 'D' as a single token. This evaluation method is not compatible with this tokenizer.")
        model.train()
        return 0.0

    for subject in tqdm(subjects, desc="Evaluating MMLU Subjects"):
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
        except Exception:
            try:
                dataset = load_dataset("cais/mmlu", subject, split="validation", trust_remote_code=True)
            except Exception as e_val:
                print(f"Warning: Could not load test or validation split for {subject}. Skipping. Error: {e_val}")
                continue

        correct_predictions, total_predictions = 0, 0
        for example in tqdm(dataset, desc=f"Subject: {subject}", leave=False):
            prompt = format_prompt(example, subject)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                last_token_logits = outputs.logits[:, -1, :]
            
            choice_logits = last_token_logits[:, choice_tokens].squeeze()
            prediction = torch.argmax(choice_logits).item()

            if prediction == example['answer']:
                correct_predictions += 1
            total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        all_accuracies[subject] = accuracy

    avg_accuracy = np.mean(list(all_accuracies.values())) if all_accuracies else 0.0
    model.train()  # IMPORTANT: Set model back to training mode
    return avg_accuracy

def calculate_sequence_loss(model, tokenizer, text, device):
    """
    Calculates the cross-entropy loss for a given sequence of text.
    Lower loss indicates higher probability under the model.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings).to(device)
    input_ids = inputs["input_ids"]
    
    # Labels are the same as input_ids, but shifted
    # The loss function in the model will handle the shifting internally
    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
    
    return loss.item()


def run_piqa_evaluation(model, tokenizer, device):
    """
    Runs PIQA evaluation on a given model and tokenizer.

    Args:
        model: The Hugging Face model to evaluate.
        tokenizer: The tokenizer associated with the model.
        device: The torch device to run on ('cuda' or 'cpu').

    Returns:
        float: The accuracy on the PIQA validation set.
    """
    model.eval()
    print("\n--- Running PIQA Evaluation ---")
    
    try:
        # Using the validation split as the test set labels are not public.
        dataset = load_dataset("baber/piqa", split="validation", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading PIQA dataset: {e}. Skipping evaluation.")
        model.train()
        return 0.0

    correct_predictions = 0
    total_predictions = 0

    for example in tqdm(dataset, desc="Evaluating PIQA"):
        goal = example['goal']
        
        # Calculate loss for both solutions
        # A lower loss means the model finds the sequence more plausible.
        loss1 = calculate_sequence_loss(model, tokenizer, f"{goal} {example['sol1']}", device)
        loss2 = calculate_sequence_loss(model, tokenizer, f"{goal} {example['sol2']}", device)

        # Predict the solution with the lower loss
        prediction = 0 if loss1 < loss2 else 1
        
        if prediction == example['label']:
            correct_predictions += 1
        total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    print(f"--- PIQA Eval complete. Accuracy: {accuracy:.4f} ---")
    
    model.train() # IMPORTANT: Set model back to training mode
    return accuracy

# --- Standalone Execution Logic ---
def evaluate_from_path(args):
    """Loads a model from a path and runs the MMLU evaluation."""
    print(f"Loading model and tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(args.device)
    
    avg_accuracy = run_mmlu_evaluation(model, tokenizer, args.device, args.limit_subjects)

    print(f"\n--- MMLU Evaluation Summary ---")
    print(f"Average Accuracy across {len(get_dataset_config_names('cais/mmlu')) if args.limit_subjects <= 0 else args.limit_subjects} subjects: {avg_accuracy:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on the MMLU benchmark.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model checkpoint directory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run evaluation on.")
    parser.add_argument("--limit_subjects", type=int, default=-1, help="Limit evaluation to the first N subjects for a quick test.")
    args = parser.parse_args()
    evaluate_from_path(args)

if __name__ == "__main__":
    main()

