import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
except ImportError:
    WordCloud = None
    plt = None

from src.model_client import LLMClient
from src.scorer import exact_match_scorer
from src.neural_scorer import NeuralScorer

# Set page config
st.set_page_config(
    page_title="LLM Evaluator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Dark Theme CSS ---
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    
    /* Hero Title */
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .hero-subtitle {
        color: #a0aec0;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out 0.2s both;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.4); }
        50% { box-shadow: 0 0 40px rgba(102, 126, 234, 0.6); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
    }
    
    /* Metric Cards */
    .metric-container {
        display: flex;
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        color: #a0aec0;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #e2e8f0 !important;
    }
    
    .sidebar-header {
        color: #667eea;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(102, 126, 234, 0.4);
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.8);
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        background-size: 200% 100%;
        animation: shimmer 2s linear infinite;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 0.5rem;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #a0aec0;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        color: #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Alerts */
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
    }
    
    /* Success message */
    .success-banner {
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.2) 0%, rgba(56, 178, 172, 0.2) 100%);
        border: 1px solid rgba(72, 187, 120, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #68d391;
        font-weight: 500;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #e2e8f0;
    }
    
    .stDownloadButton > button:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    /* Section headers */
    h2, h3 {
        color: #e2e8f0 !important;
        font-weight: 600;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown('<h1 class="hero-title">🧠 LLM Evaluator</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Evaluate your language models with precision and style</p>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Settings</div>', unsafe_allow_html=True)
    
    scorer_type = st.radio(
        "Evaluation Method",
        options=["Exact Match", "Neural Scorer"],
        help="Exact Match: Strict comparison\nNeural Scorer: AI-based semantic evaluation"
    )
    
    st.markdown("---")
    
    # Concurrency setting
    concurrency = st.slider("Concurrency", min_value=5, max_value=50, value=20, 
                           help="Number of parallel API requests")
    
    st.markdown("---")
    st.markdown('<div class="sidebar-header">📜 History</div>', unsafe_allow_html=True)
    
    # Session state for history
    if 'eval_history' not in st.session_state:
        st.session_state.eval_history = []
    
    if st.session_state.eval_history:
        for i, h in enumerate(st.session_state.eval_history[-5:][::-1]):
            st.markdown(f"**{h['timestamp']}** - {h['accuracy']:.1%} ({h['count']} items)")
    else:
        st.caption("No evaluations yet")

# --- Initialize Resources ---
@st.cache_resource
def get_client():
    return LLMClient()

@st.cache_resource
def get_neural_scorer():
    model_path = 'neural_scorer_model'
    if os.path.exists(model_path):
        try:
            return NeuralScorer(model_name=model_path)
        except Exception:
            return None
    return None

neural_scorer = None
if scorer_type == "Neural Scorer":
    neural_scorer = get_neural_scorer()
    if neural_scorer is None:
        st.sidebar.warning("⚠️ Neural model not found. Using Exact Match.")
        scorer_type = "Exact Match"
    else:
        st.sidebar.success("✅ Neural Scorer loaded")

# --- Helper Functions ---
def infer_category(question):
    q = question.lower()
    if any(k in q for k in ['x', 'y', 'solve for']):
        return 'Algebra'
    elif any(k in q for k in ['rectangle', 'square', 'perimeter', 'area']):
        return 'Geometry'
    elif any(k in q for k in ['train', 'john', 'apples', 'cost']):
        return 'Word Problem'
    elif any(k in q for k in ['sequence', 'next number']):
        return 'Logic'
    return 'Arithmetic'

async def evaluate_item(client, item, sem, scorer_type, neural_scorer_instance):
    async with sem:
        try:
            response = await client.get_response(item['question'])
            is_correct = False
            confidence = 1.0
            
            if scorer_type == "Neural Scorer" and neural_scorer_instance:
                is_correct, confidence = neural_scorer_instance.predict(
                    item['question'], item['answer'], response
                )
            else:
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
            }
        except Exception as e:
            return {
                "id": item.get('id', 'unknown'),
                "error": str(e),
                "is_correct": False,
                "response": "",
                "confidence": 0.0,
            }

async def run_evaluation(items, scorer_type_selected, neural_scorer_obj, concurrency_limit):
    client = get_client()
    sem = asyncio.Semaphore(concurrency_limit)
    tasks = [evaluate_item(client, item, sem, scorer_type_selected, neural_scorer_obj) for item in items]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    completed = 0
    total = len(items)
    results = []
    
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        completed += 1
        progress_bar.progress(completed / total)
        status_text.caption(f"Processing... {completed}/{total}")
    
    status_text.empty()
    return results

# --- Plotting Functions ---
DARK_TEMPLATE = "plotly_dark"
COLOR_PALETTE = ['#667eea', '#764ba2', '#f093fb', '#48bb78', '#ed8936', '#4fd1c5']

def plot_radar_chart(df):
    cat_accuracy = df.groupby('category')['is_correct'].mean().reset_index()
    cat_accuracy.columns = ['category', 'accuracy']
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(cat_accuracy['accuracy']) + [cat_accuracy['accuracy'].iloc[0]],
        theta=list(cat_accuracy['category']) + [cat_accuracy['category'].iloc[0]],
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=2),
        name='Accuracy'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.1)'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        showlegend=False,
        margin=dict(t=40, b=40, l=60, r=60)
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_accuracy_pie(correct, incorrect):
    fig = go.Figure(data=[go.Pie(
        labels=['Correct', 'Incorrect'],
        values=[correct, incorrect],
        hole=0.6,
        marker=dict(colors=['#48bb78', '#e53e3e']),
        textinfo='percent+label',
        textfont=dict(size=14, color='white')
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        annotations=[dict(text='Overall', x=0.5, y=0.5, font_size=16, showarrow=False, font_color='#a0aec0')]
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_category_bar(df):
    cat_stats = df.groupby('category').agg(
        correct=('is_correct', 'sum'),
        total=('is_correct', 'count')
    ).reset_index()
    cat_stats['accuracy'] = cat_stats['correct'] / cat_stats['total']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cat_stats['category'],
        y=cat_stats['accuracy'],
        marker=dict(
            color=cat_stats['accuracy'],
            colorscale=[[0, '#e53e3e'], [0.5, '#ed8936'], [1, '#48bb78']],
            line=dict(width=0)
        ),
        text=[f"{acc:.0%}" for acc in cat_stats['accuracy']],
        textposition='outside',
        textfont=dict(color='#e2e8f0')
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[0, 1.15]),
        margin=dict(t=40, b=40),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_length_distribution(df):
    df = df.copy()
    df['length'] = df['response'].astype(str).apply(len)
    df['Status'] = df['is_correct'].apply(lambda x: 'Correct' if x else 'Incorrect')
    
    fig = px.histogram(
        df, x='length', color='Status', nbins=25,
        color_discrete_map={'Correct': '#48bb78', 'Incorrect': '#e53e3e'},
        opacity=0.8, barmode='overlay',
        template=DARK_TEMPLATE
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='Response Length (chars)',
        yaxis_title='Count',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_confidence_distribution(df):
    if 'confidence' not in df.columns:
        return
    
    df = df.copy()
    df['Status'] = df['is_correct'].apply(lambda x: 'Correct' if x else 'Incorrect')
    
    fig = px.histogram(
        df, x='confidence', color='Status', nbins=20,
        color_discrete_map={'Correct': '#48bb78', 'Incorrect': '#e53e3e'},
        opacity=0.8, barmode='overlay',
        template=DARK_TEMPLATE
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='Confidence Score',
        yaxis_title='Count'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_error_wordcloud(df):
    if WordCloud is None:
        st.warning("Install `wordcloud` for this visualization")
        return
    
    error_df = df[~df['is_correct']]
    if len(error_df) == 0:
        st.info("🎉 No errors to analyze!")
        return
    
    text = " ".join(error_df['response'].astype(str))
    wordcloud = WordCloud(
        width=800, height=400, 
        background_color='#1a1a2e',
        colormap='plasma',
        max_words=100
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1a2e')
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# --- Main App ---
st.markdown("---")

uploaded_file = st.file_uploader("📁 Upload your test dataset", type="jsonl", 
                                   help="JSONL format with 'question' and 'answer' fields")

if uploaded_file is not None:
    lines = uploaded_file.getvalue().decode("utf-8").strip().splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    
    st.info(f"📊 Loaded **{len(items)}** test cases")
    
    col_btn, col_spacer = st.columns([1, 3])
    with col_btn:
        start_btn = st.button("🚀 Start Evaluation", type="primary", use_container_width=True)
    
    if start_btn:
        with st.spinner(""):
            try:
                results = asyncio.run(run_evaluation(items, scorer_type, neural_scorer, concurrency))
                df = pd.DataFrame(results)
                df['is_correct'] = df['is_correct'].astype(bool)
                accuracy = df['is_correct'].mean()
                correct_count = int(df['is_correct'].sum())
                incorrect_count = len(df) - correct_count
                
                # Add to history
                from datetime import datetime
                st.session_state.eval_history.append({
                    'timestamp': datetime.now().strftime('%H:%M'),
                    'accuracy': accuracy,
                    'count': len(df)
                })
                
                # Success banner
                st.markdown(f'<div class="success-banner">✅ Evaluation Complete!</div>', unsafe_allow_html=True)
                
                # Metrics with custom HTML
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-icon">📝</div>
                        <div class="metric-value">{len(df)}</div>
                        <div class="metric-label">Total Questions</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-icon">✅</div>
                        <div class="metric-value">{correct_count}</div>
                        <div class="metric-label">Correct</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-icon">❌</div>
                        <div class="metric-value">{incorrect_count}</div>
                        <div class="metric-label">Incorrect</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-icon">🎯</div>
                        <div class="metric-value">{accuracy:.1%}</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Tabs
                tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔬 Analysis", "📋 Data"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("##### Accuracy by Category")
                        if df['category'].nunique() >= 2:
                            plot_radar_chart(df)
                        else:
                            plot_category_bar(df)
                    with col2:
                        st.markdown("##### Overall Results")
                        plot_accuracy_pie(correct_count, incorrect_count)
                
                with tab2:
                    col3, col4 = st.columns(2)
                    with col3:
                        st.markdown("##### Response Length")
                        plot_length_distribution(df)
                    with col4:
                        st.markdown("##### Error Patterns")
                        plot_error_wordcloud(df)
                    
                    if scorer_type == "Neural Scorer":
                        st.markdown("##### Confidence Distribution")
                        plot_confidence_distribution(df)
                
                with tab3:
                    st.dataframe(
                        df[['id', 'category', 'question', 'answer', 'response', 'is_correct', 'confidence']],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.markdown("---")
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download CSV", csv_data, "results.csv", "text/csv", use_container_width=True)
                    with col_d2:
                        jsonl_data = df.to_json(orient="records", lines=True).encode('utf-8')
                        st.download_button("📥 Download JSONL", jsonl_data, "results.jsonl", "application/json", use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
