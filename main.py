import json
import asyncio
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from src.model_client import LLMClient
from src.scorer import exact_match_scorer

def visualize_results(results, detailed_data, save_path=None):
    """
    Generate visualization charts for evaluation results.
    
    Args:
        results: List of boolean values indicating correctness
        detailed_data: List of dicts with question details
        save_path: Optional file path to save figure
    """
    
    # Set professional style
    plt.style.use('ggplot')
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Accuracy Pie Chart
    correct_count = sum(results)
    incorrect_count = len(results) - correct_count
    
    colors = ['#4CAF50', '#F44336']  # Green for correct, red for incorrect
    explode = (0.05, 0)  # Slightly separate the correct slice
    
    ax1.pie([correct_count, incorrect_count], 
            labels=['Correct', 'Incorrect'],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=explode,
            shadow=True)
    ax1.set_title(f'Overall Accuracy: {correct_count}/{len(results)} ({100*correct_count/len(results):.1f}%)', 
                  fontsize=14, fontweight='bold')
    
    # Subplot 2: Per-Question Bar Chart
    question_ids = [item['id'] for item in detailed_data]
    bar_colors = ['#4CAF50' if r else '#F44336' for r in results]
    
    ax2.bar(question_ids, [1]*len(question_ids), color=bar_colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Question ID', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Result', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Question Results', fontsize=14, fontweight='bold')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Incorrect', 'Correct'])
    ax2.grid(axis='x', alpha=0.3)
    
    # Add summary text
    accuracy = correct_count / len(results)
    fig.suptitle(f'LLM Evaluation Results - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()

async def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation Pipeline")
    parser.add_argument("--data_path", type=str, default="data/test_cases.jsonl")
    parser.add_argument("--visualize", action="store_true", help="Display visualization charts interactively")
    parser.add_argument("--save-viz", type=str, default=None,
                        help="Save visualization to specified file path (e.g., results.png)")
    parser.add_argument("--neural-scorer", action="store_true", help="Use trained Neural Scorer model")
    parser.add_argument("--scorer-model", type=str, default="neural_scorer_model", help="Path to trained scorer model")
    args = parser.parse_args()

    client = LLMClient()
    
    # Initialize scorer
    neural_scorer_model = None
    if args.neural_scorer:
        try:
            from src.neural_scorer import NeuralScorer
            print(f"Loading Neural Scorer from {args.scorer_model}...")
            neural_scorer_model = NeuralScorer(model_name=args.scorer_model)
        except Exception as e:
            print(f"Failed to load Neural Scorer: {e}")
            print("Falling back to Exact Match Scorer.")

    results = []
    detailed_data = []

    print(f"Loading data from {args.data_path}...")
    
    # Read all data first
    items = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
            
    # Semaphore for rate limiting (e.g., 20 concurrent requests)
    # 1000 requests/minute = ~16.6 requests/second. 
    # With generic network latency, 20-50 concurrent tasks is a safe starting point.
    sem = asyncio.Semaphore(20)

    async def evaluate_item(item):
        async with sem:
            print(f"Evaluating ID {item['id']}...")
            try:
                response = await client.get_response(item['question'])
                
                # Score the response
                if neural_scorer_model:
                     # Neural Scorer
                    is_correct, conf = neural_scorer_model.predict(item['question'], item['answer'], response)
                    # We might want to store confidence too, but for now just bool
                else:
                    # Default Exact Match
                    extracted = client.extract_final_answer(response) 
                    is_correct = exact_match_scorer(extracted, item['answer'])
                
                return is_correct, item
            except Exception as e:
                print(f"Error evaluating ID {item['id']}: {e}")
                return False, item

    print(f"Starting concurrent evaluation of {len(items)} items...")
    tasks = [evaluate_item(item) for item in items]
    
    # Run all tasks
    evaluation_results = await asyncio.gather(*tasks)
    
    # Unpack results
    for is_correct, item in evaluation_results:
        results.append(is_correct)
        detailed_data.append(item)

    # Calculate accuracy
    accuracy = sum(results) / len(results) if results else 0
    print("-" * 30)
    print(f"Evaluation Finished!")
    print(f"Final Accuracy: {accuracy:.2%}")
    
    # Generate visualization if requested
    if args.visualize or args.save_viz:
        print("\nGenerating visualization...")
        visualize_results(results, detailed_data, args.save_viz)

if __name__ == "__main__":
    asyncio.run(main())