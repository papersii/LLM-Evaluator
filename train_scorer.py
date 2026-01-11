import json
from src.neural_scorer import NeuralScorer
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Train Neural Scorer")
    # For now, we will create dummy data if file not found, or use the test cases if they have labels.
    # The existing test_cases.jsonl have 'answer' but not 'label' (correct/incorrect) for a (question, answer, response) triplet.
    # To properly train, we need to generate synthetic data:
    # 1. Positive samples: Response matches Answer
    # 2. Negative samples: Response is wrong/different
    
    args = parser.parse_args()
    
    print("Generating synthetic training data...")
    train_data = []
    
    # 1. Math examples
    train_data.append({
        "question": "What is 2+2?", "answer": "4", "response": "The answer is 4", "label": 1
    })
    train_data.append({
        "question": "What is 2+2?", "answer": "4", "response": "The answer is 5", "label": 0
    })
    
    # 2. Text examples
    train_data.append({
        "question": "Capital of France?", "answer": "Paris", "response": "Paris is the capital", "label": 1
    })
    train_data.append({
        "question": "Capital of France?", "answer": "Paris", "response": "It is London", "label": 0
    })
    
    # Add more synthetic data for stability
    for i in range(10):
        train_data.append({
            "question": f"What is {i}+{i}?", "answer": f"{i+i}", "response": f"{i+i}", "label": 1
        })
        train_data.append({
            "question": f"What is {i}+{i}?", "answer": f"{i+i}", "response": f"{i+i+1}", "label": 0
        })

    scorer = NeuralScorer(model_name='bert-base-uncased')
    scorer.train(train_data, epochs=3)
    
    print("Training complete. Verifying inference...")
    is_correct, conf = scorer.predict("What is 10+10?", "20", "The answer is 20")
    print(f"Test Positive: Correct={is_correct}, Conf={conf:.4f}")
    
    is_correct, conf = scorer.predict("What is 10+10?", "20", "The answer is 99")
    print(f"Test Negative: Correct={is_correct}, Conf={conf:.4f}")

if __name__ == "__main__":
    main()
