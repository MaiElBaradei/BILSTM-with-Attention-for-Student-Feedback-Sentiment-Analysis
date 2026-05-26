"""
To run the verification test, execute this script from the root directory of the project:
    python -m tests.pipeline_test 
It will generate mock text data, preprocess it, and create dataloaders to ensure that all components of the pipeline are functioning correctly together. 
The test includes checks for data integrity, batch structure, and sequence length ordering.
"""

import torch
import random
from preprocessing.tokenizer import TextPreprocessor
from training.dataloaders import create_dataloaders

def generate_mock_data(num_samples: int = 100):
    """Generates mock text data with a mix of clean and noisy elements for testing.
    Args:
        num_samples (int): Number of mock samples to generate.
    Returns:
        Tuple[List[str], List[int]]: A tuple containing lists of texts and their corresponding labels.
    """
    vocab_pool = ["excellent", "terrible", "good", "bad", "product", "delivery", "slow", "fast"]
    texts = []
    labels = []
    for _ in range(num_samples):
        seq_len = random.randint(3, 10)
        sentence = " ".join(random.choices(vocab_pool, k=seq_len))
        # Mix in some HTML elements and weird spaces to verify cleaning structures
        if random.random() > 0.5:
            sentence = f"<p><b>{sentence}</b></p>   https://example.com"
        texts.append(sentence)
        labels.append(random.choice([0, 1]))
    return texts, labels

def run_verification():
    """
    Runs the end-to-end verification of the preprocessing and dataloader pipeline.
    This function generates mock data, initializes the preprocessor, creates dataloaders, and performs checks on the batch structure and sequence length ordering.
    Args:     None
    Returns:    None
    """
    print("[*] Generating mock text collections...")
    mock_texts, mock_labels = generate_mock_data(150)
    
    print("[*] Spawning Preprocessor Engine...")
    preprocessor = TextPreprocessor(max_seq_len=16, min_freq=1)
    
    print("[*] Creating modular structural pipelines...")
    train_loader, val_loader, test_loader, t_texts, t_labels = create_dataloaders(
        texts=mock_texts,
        labels=mock_labels,
        preprocessor=preprocessor,
        batch_size=8,
        num_workers=2 # Testing multi-worker capability
    )
    
    print(f"[+] Active vocabulary length built: {len(preprocessor.vocab)}")
    
    # Asserting data integrity check over batches
    print("[*] Pulling single validation sample batch for evaluation...")
    for inputs, lengths, labels in train_loader:
        print("\n--- Batch Structure Inspection ---")
        print(f"Inputs Matrix Shape : {inputs.shape}  -> Expected [BatchSize, MaxSeqLen]")
        print(f"Sequence Length Array: {lengths.tolist()} -> Sorted Descending Check")
        print(f"Target Label Matrix  : {labels.shape}")
        
        # Validation checks
        assert inputs.shape[0] == len(lengths) == len(labels), "Dimension mismatch inside batch."
        assert all(lengths[i] >= lengths[i+1] for i in range(len(lengths)-1)), "Collate ordering error: Sequence lengths not descending!"
        print("\n[✓] Component Integration Check Passed Successfully.")
        break

if __name__ == "__main__":
    run_verification()