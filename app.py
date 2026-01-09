import streamlit as st
import asyncio
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

from src.model_client import LLMClient
from src.scorer import exact_match_scorer

# Set page config
st.set_page_config(
    page_title="LLM Evaluator Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 LLM Evaluator Dashboard")
st.markdown("Upload a test dataset (JSONL) to evaluate your LLM model and visualize the results.")

# Initialize client
@st.cache_resource
def get_client():
    return LLMClient()

def infer_category(question):
    question = question.lower()
    if any(k in question for k in ['x', 'y', 'solve for']):
        return 'Algebra'
    elif any(k in question for k in ['rectangle', 'square', 'perimeter', 'area']):
        return 'Geometry'
    elif any(k in question for k in ['train', 'john', 'apples', 'cost']):
        return 'Word Problem'
    elif any(k in question for k in ['sequence', 'next number']):
        return 'Logic'
    else:
        return 'Arithmetic'

async def evaluate_item(client, item, sem):
    async with sem:
        try:
            response = await client.get_response(item['question'])
            is_correct = exact_match_scorer(response, item['answer'])
            return {
                "id": item.get('id', 'unknown'),
                "category": item.get('category', infer_category(item['question'])),
                "question": item['question'],
                "answer": item['answer'],
                "response": response,
                "is_correct": is_correct
            }
        except Exception as e:
            return {
                "id": item.get('id', 'unknown'),
                "error": str(e),
                "is_correct": False,
                "response": ""
            }

async def run_evaluation(items):
    client = get_client()
    sem = asyncio.Semaphore(20) # Rate limit
    
    tasks = [evaluate_item(client, item, sem) for item in items]
    
    # Progress bar
    progress_bar = st.progress(0)
    completed_count = 0
    total = len(items)
    results = []
    
    # Run tasks and update progress
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        completed_count += 1
        progress_bar.progress(completed_count / total)
        
    return results

def plot_radar_chart(df):
    # Group by category
    categories = df['category'].unique()
    
    # Calculate accuracy per category
    cat_accuracy = df.groupby('category')['is_correct'].mean()
    
    # If less than 3 categories, add dummy ones for a proper polygon
    labels = list(cat_accuracy.index)
    values = list(cat_accuracy.values)
    
    if len(labels) < 3:
        # Create a "pseudo-radar" or just warn
        # For better visuals, let's force a triangle if needed or just handle it
        pass
        
    # Number of variables
    N = len(labels)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] # Close the circle
    
    values += values[:1] # Close the circle
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], labels)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)
    
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)
    
    st.pyplot(fig)

def plot_length_distribution(df):
    ft_size = 12
    # Calculate lengths (safely handle non-string values)
    df['length'] = df['response'].astype(str).apply(len)
    
    correct_lengths = df[df['is_correct']]['length']
    incorrect_lengths = df[~df['is_correct']]['length']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    ax.hist(correct_lengths, bins=20, alpha=0.7, label='Correct', color='green', edgecolor='black')
    ax.hist(incorrect_lengths, bins=20, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    
    ax.set_title('Response Length Distribution', fontsize=ft_size)
    ax.set_xlabel('Response Length (chars)', fontsize=ft_size)
    ax.set_ylabel('Count', fontsize=ft_size)
    ax.legend()
    
    st.pyplot(fig)

def plot_error_wordcloud(df):
    if WordCloud is None:
        st.warning("WordCloud library not installed. Please install 'wordcloud'.")
        return

    error_df = df[~df['is_correct']]
    
    if len(error_df) == 0:
        st.info("No incorrect answers to analyze!")
        return
        
    text = " ".join(error_df['response'].astype(str))
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    st.pyplot(fig)

# File Uploader
uploaded_file = st.file_uploader("Choose a JSONL file", type="jsonl")

if uploaded_file is not None:
    # Read file
    lines = uploaded_file.getvalue().decode("utf-8").strip().splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    
    st.info(f"Loaded {len(items)} test cases.")
    
    if st.button("Start Evaluation"):
        with st.spinner("Evaluating..."):
            # Run async evaluation
            try:
                results = asyncio.run(run_evaluation(items))
                
                # Process results
                df = pd.DataFrame(results)
                accuracy = df['is_correct'].mean()
                
                st.success(f"Evaluation Complete! Final Accuracy: {accuracy:.2%}")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Questions", len(df))
                col2.metric("Correct Answers", df['is_correct'].sum())
                col3.metric("Accuracy", f"{accuracy:.2%}")
                
                # Visualization
                st.subheader("Radar Chart Analysis")
                if 'category' in df.columns and df['category'].nunique() >= 1:
                     plot_radar_chart(df)
                else:
                    st.warning("Not enough categories for Radar Chart.")
                
                # Detailed Data
                st.subheader("Detailed Results")
                st.dataframe(df)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

            # Export Results
            st.divider()
            st.subheader("📥 Export Results")
            col1, col2 = st.columns(2)
            
            # CSV Export
            csv_data = df.to_csv(index=False).encode('utf-8')
            col1.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name="evaluation_results.csv",
                mime="text/csv",
            )
            
            # JSONL Export
            jsonl_data = df.to_json(orient="records", lines=True).encode('utf-8')
            col2.download_button(
                label="Download as JSONL",
                data=jsonl_data,
                file_name="evaluation_results.jsonl",
                mime="application/json",
            )
            
            # Advanced Visualization
            st.divider()
            st.subheader("🔬 Advanced Analysis")
            
            tab1, tab2 = st.tabs(["Response Length", "Error Patterns (WordCloud)"])
            
            with tab1:
                plot_length_distribution(df)
                
            with tab2:
                plot_error_wordcloud(df)

