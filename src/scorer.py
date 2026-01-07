import re

def exact_match_scorer(prediction, ground_truth):
    """
    Extract numerical answer from model response using regex and compare with ground truth.
    Supports multiple common formats: \boxed{answer}, "The answer is X", final numbers, etc.
    """
    if prediction is None:
        return False
    
    # Try multiple patterns in priority order
    patterns = [
        r'\\boxed\{([^}]+)\}',              # LaTeX: \boxed{7}
        r'#### ([^\n]+)',                   # GSM8K format: #### 42
        r'The answer is[:\s]+([^\n]+)',     # "The answer is: 7"
        r'Answer[:\s]+([^\n]+)',            # "Answer: 7"
        r'答案是[：:\s]*([^\n。]+)',          # Chinese format: "答案是：7"
        r'(?:^|[^\d\.])(\d+(?:\.\d+)?)(?=[^\d\.]*$)' # Last number (int or float) in text
    ]
    
    for pattern in patterns:
        match = re.search(pattern, prediction, re.IGNORECASE)
        if match:
            extracted_answer = match.group(1).strip()
            # Remove spaces and symbols
            extracted_answer = re.sub(r'[^\d\.]', '', extracted_answer)
            try:
                # Attempt float comparison
                if float(extracted_answer) == float(ground_truth):
                    return True
            except (ValueError, TypeError):
                # Fallback to string comparison
                if extracted_answer and str(extracted_answer) == str(ground_truth):
                    return True
    
    return False