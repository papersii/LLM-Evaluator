import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt # Keep matplotlib for wordcloud
except ImportError:
    WordCloud = None
    plt = None

from src.model_client import LLMClient
from src.scorer import exact_match_scorer
from src.neural_scorer import NeuralScorer

# Set page config
st.set_page_config(
    page_title="LLM Evaluator Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .stMetric label {
        color: #666;
    }
    h1, h2, h3 {
        color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 LLM Evaluator Dashboard")
st.markdown("Upload a test dataset (JSONL) to evaluate your LLM model and visualize the results.")

# --- Sidebar ---
st.sidebar.header("⚙️ Evaluation Settings")

# Scorer Selection
scorer_type = st.sidebar.radio(
    "Select Scorer",
    options=["Exact Match", "Neural Scorer"],
    help="Exact Match: Strict string comparison.\nNeural Scorer: AI-based semantic evaluation (requires trained model)."
)

# Initialize Resources
@st.cache_resource
def get_client():
    return LLMClient()

@st.cache_resource
def get_neural_scorer():
    model_path = 'neural_scorer_model'
    if os.path.exists(model_path):
        try:
            return NeuralScorer(model_name=model_path)
        except Exception as e:
            st.error(f"Failed to load Neural Scorer: {e}")
            return None
    else:
        return None

# Check Neural Scorer Availability
neural_scorer = None
if scorer_type == "Neural Scorer":
    neural_scorer = get_neural_scorer()
    if neural_scorer is None:
        st.sidebar.warning("⚠️ Neural Scorer model not found in 'neural_scorer_model/'. Falling back to Exact Match.")
        scorer_type = "Exact Match"
    else:
        st.sidebar.success("✅ Neural Scorer Loaded")

# --- Helper Functions ---

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

async def evaluate_item(client, item, sem, scorer_type, neural_scorer_instance):
    async with sem:
        try:
            response = await client.get_response(item['question'])
            
            is_correct = False
            confidence = 1.0
            
            if scorer_type == "Neural Scorer" and neural_scorer_instance:
                # Returns (is_correct, confidence)
                is_correct, confidence = neural_scorer_instance.predict(
                    item['question'], item['answer'], response
                )
            else:
                # Exact Match
                is_correct = exact_match_scorer(response, item['answer'])
                confidence = 1.0 if is_correct else 0.0

            return {
                "id": item.get('id', 'unknown'),
                "category": item.get('category', infer_category(item['question'])),
                "question": item['question'],
                "answer": item['answer'],
                "response": response,
                "is_correct": is_correct,
                "confidence": confidence,
                "scorer": scorer_type
            }
        except Exception as e:
            return {
                "id": item.get('id', 'unknown'),
                "error": str(e),
                "is_correct": False,
                "response": "",
                "confidence": 0.0,
                "scorer": scorer_type
            }

async def run_evaluation(items, scorer_type_selected, neural_scorer_obj):
    client = get_client()
    sem = asyncio.Semaphore(20) # Rate limit
    
    tasks = [evaluate_item(client, item, sem, scorer_type_selected, neural_scorer_obj) for item in items]
    
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

# --- Plotting Functions (Plotly) ---

def plot_radar_chart(df):
    # Group by category
    cat_accuracy = df.groupby('category')['is_correct'].mean().reset_index()
    cat_accuracy.columns = ['category', 'accuracy']
    
    fig = px.line_polar(
        cat_accuracy,
        r='accuracy',
        theta='category',
        line_close=True,
        range_r=[0, 1],
        title='Accuracy by Category',
        markers=True,
        template="plotly_white"
    )
    fig.update_traces(fill='toself', line_color='#4C78A8')
    st.plotly_chart(fig, use_container_width=True)

def plot_length_distribution(df):
    df['length'] = df['response'].astype(str).apply(len) if 'response' in df else 0
    df['Status'] = df['is_correct'].apply(lambda x: 'Correct' if x else 'Incorrect')
    
    fig = px.histogram(
        df, 
        x='length', 
        color='Status',
        nbins=20,
        title='Response Length Distribution',
        color_discrete_map={'Correct': '#00CC96', 'Incorrect': '#EF553B'},
        opacity=0.7,
        template="plotly_white",
        labels={'length': 'Response Length (chars)'}
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_confidence_distribution(df):
    if 'confidence' not in df.columns:
        return
        
    fig = px.histogram(
        df,
        x='confidence',
        color='is_correct',
        nbins=20,
        title='Confidence Score Distribution',
        labels={'confidence': 'Confidence Score'},
        template="plotly_white",
         color_discrete_map={True: '#00CC96', False: '#EF553B'}
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_error_wordcloud(df):
    if WordCloud is None:
        st.warning("WordCloud library not installed.")
        return

    error_df = df[~df['is_correct']]
    
    if len(error_df) == 0:
        st.info("No incorrect answers to analyze!")
        return
        
    text = " ".join(error_df['response'].astype(str))
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Use matplotlib for wordcloud as it renders an image
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    st.pyplot(fig)

# --- Main App ---

uploaded_file = st.file_uploader("Choose a JSONL file", type="jsonl")

if uploaded_file is not None:
    # Read file
    lines = uploaded_file.getvalue().decode("utf-8").strip().splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    
    st.info(f"Loaded {len(items)} test cases.")
    
    if st.button("Start Evaluation", type="primary"):
        with st.spinner("Evaluating..."):
            # Run async evaluation
            try:
                results = asyncio.run(run_evaluation(items, scorer_type, neural_scorer))
                
                # Process results
                df = pd.DataFrame(results)
                
                # Handle boolean conversion for proper display
                df['is_correct'] = df['is_correct'].astype(bool)
                
                accuracy = df['is_correct'].mean()
                
                st.success("Evaluation Complete!")
                
                # --- Metrics Section ---
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Questions", len(df))
                col2.metric("Correct Answers", int(df['is_correct'].sum()))
                col3.metric("Accuracy", f"{accuracy:.2%}")
                
                st.divider()

                # --- Visualization Tabs ---
                tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔬 Detailed Analysis", "💾 Raw Data"])
                
                with tab1:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if 'category' in df.columns and df['category'].nunique() >= 1:
                            plot_radar_chart(df)
                        else:
                            st.warning("Not enough categories for Radar Chart.")
                    with col_b:
                        # Pie chart for overall accuracy
                        fig_pie = px.pie(
                            names=['Correct', 'Incorrect'],
                            values=[df['is_correct'].sum(), (~df['is_correct']).sum()],
                            title="Overall Accuracy",
                            color_discrete_sequence=['#00CC96', '#EF553B'],
                            hole=0.4
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                with tab2:
                    col_c, col_d = st.columns(2)
                    with col_c:
                         plot_length_distribution(df)
                    with col_d:
                        plot_error_wordcloud(df)
                    
                    if scorer_type == "Neural Scorer":
                         plot_confidence_distribution(df)

                with tab3:
                    st.dataframe(df, use_container_width=True)
                    
                    # Export Buttons
                    col_exp1, col_exp2 = st.columns(2)
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    col_exp1.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="evaluation_results.csv",
                        mime="text/csv",
                    )
                    
                    jsonl_data = df.to_json(orient="records", lines=True).encode('utf-8')
                    col_exp2.download_button(
                        label="Download JSONL",
                        data=jsonl_data,
                        file_name="evaluation_results.jsonl",
                        mime="application/json",
                    )
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                # Print proper stack trace in logs if needed, but for UI just show error
                import traceback
                st.text(traceback.format_exc())
