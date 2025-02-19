import re

def extract_answer(text):
    """Extract the final numerical answer from the model's response."""
    # Look for the last number in the text
    numbers = re.findall(r'-?\d*\.?\d+', text)
    return float(numbers[-1]) if numbers else None

def compute_gsm8k_metrics(predictions, references):
    """
    Compute GSM8K accuracy metrics.
    Args:
        predictions: List of model outputs
        references: List of reference outputs
    Returns:
        dict: Metrics including accuracy and individual results
    """
    correct = 0
    results = []
    
    for pred, ref in zip(predictions, references):
        pred_answer = extract_answer(pred)
        ref_answer = extract_answer(ref)
        
        is_correct = pred_answer == ref_answer if pred_answer is not None else False
        if is_correct:
            correct += 1
            
        results.append({
            "prediction": pred,
            "reference": ref,
            "predicted_answer": pred_answer,
            "reference_answer": ref_answer,
            "is_correct": is_correct
        })
    
    metrics = {
        "accuracy": correct / len(predictions) if predictions else 0,
        "results": results
    }
    
    return metrics 