import re

def extract_answer_last_number(text):
    """Extract the final numerical answer from the model's response using last number method."""
    numbers = re.findall(r'-?\d*\.?\d+', text)
    return float(numbers[-1]) if numbers else None

def extract_answer_hash(text):
    """Extract the answer that follows '####' in the text."""
    pattern = r'####\s*(-?\d*\.?\d+)'
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None

def compute_gsm8k_metrics(predictions, references, method="hash"):
    """
    Compute GSM8K accuracy metrics.
    Args:
        predictions: List of model outputs
        references: List of reference outputs
        method: Extraction method ("hash" or "last_number")
    Returns:
        dict: Metrics including accuracy and individual results
    """
    extract_fn = extract_answer_hash
    correct = 0
    results = []
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred_answer = extract_fn(pred)
        ref_answer = extract_fn(ref)
        
        # Add debug info if extraction fails
        # if pred_answer is None:
        #     print(f"\nWarning: Could not extract prediction from example {i}")
        #     print(f"Prediction text: {pred[:200]}...")
        #     print(f"Using method: {method}")
        # if ref_answer is None:
        #     print(f"\nWarning: Could not extract reference from example {i}")
        #     print(f"Reference text: {ref[:200]}...")
        #     print(f"Using method: {method}")
        
        is_correct = pred_answer == ref_answer if (pred_answer is not None and ref_answer is not None) else False
        if is_correct:
            correct += 1
            
        results.append({
            "prediction": pred,
            "reference": ref,
            "predicted_answer": pred_answer,
            "reference_answer": ref_answer,
            "is_correct": is_correct,
            "extraction_method": method
        })
    
    metrics = {
        "accuracy": correct / len(predictions) if predictions else 0,
        "results": results,
        "total_examples": len(predictions),
        "correct_count": correct,
        "failed_extractions": sum(1 for r in results if r["predicted_answer"] is None or r["reference_answer"] is None),
        "extraction_method": method
    }
    
    return metrics 