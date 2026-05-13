import streamlit as st
import spacy
import plotly.express as px
import pandas as pd

@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

st.set_page_config(page_title="AI Visual Analyzer", page_icon="📊", layout="wide")

st.title("🚀 Advanced AI Skill Visualizer")
st.write("Analyze your expertise with interactive charts!")

# Layout for columns
col1, col2 = st.columns([1, 1])

with col1:
    user_input = st.text_area("Paste your AI Project details or Resume text here:", height=300)
    analyze_btn = st.button("Generate Visual Analysis")

if analyze_btn and user_input:
    with st.spinner("AI is calculating similarity..."):
        doc = nlp(user_input)
        
        categories = {
            "Machine Learning": nlp("machine learning neural networks math algorithms"),
            "Computer Vision": nlp("images opencv cameras detection vision"),
            "Natural Language Processing": nlp("text translation speech nlp spacy"),
            "Software Development": nlp("python coding sql databases backend")
        }

        names = []
        scores = []

        for name, model in categories.items():
            score = round(doc.similarity(model) * 100, 2)
            names.append(name)
            scores.append(score)

        # Create a DataFrame for the chart
        df = pd.DataFrame({"Skill": names, "Expertise %": scores})

        with col2:
            st.subheader("Your Skill Graph")
            # Creating a colorful bar chart
            fig = px.bar(df, x='Skill', y='Expertise %', color='Skill', text='Expertise %',
                         title="AI Skill Distribution", color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.success("Analysis Complete! You can see your strengths in the graph above.")
