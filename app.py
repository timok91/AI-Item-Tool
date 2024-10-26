import streamlit as st
import uuid
import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.express as px
import pandas as pd
import factor_analyzer
from factor_analyzer import FactorAnalyzer
import plotly.graph_objects as go
from anthropic import Anthropic
from item_generation.item_generator import GermanPersonalityItemGenerator

# Page config
st.set_page_config(
    page_title="AI Item-Entwicklungs-Tool",
    page_icon="assets/sapientia-favicon.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        html, body, [class*="css"], [class*="st-"] {
            font-family: Georgia, serif !important;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: Georgia, serif !important;
        }
        .stMarkdown, .stText {
            font-family: Georgia, serif !important;
        }
        .stButton button {
            font-family: Georgia, serif !important;
        }
        button[data-baseweb="tab"] {
            font-family: Georgia, serif !important;
        }
        .st-emotion-cache-1629p8f h1, h2, h3 {
            font-family: Georgia, serif !important;
        }
        .st-emotion-cache-10trblm {
            font-family: Georgia, serif !important;
        }
        .st-emotion-cache-6qob1r {
            font-family: Georgia, serif !important;
        }
    </style>
""", unsafe_allow_html=True)

# Default values
DEFAULT_CONSTRUCT = """Emotionsregulation beschreibt den Prozess, durch den Individuen das Erleben, die Intensität, die Dauer, den Zeitpunkt und den Ausdruck von aktivierten Emotionen beeinflussen. Durch Emotionsregulation können pos. und neg. Emotionen verstärkt (Verstärkung), aufrechterhalten oder abgeschwächt werden. Emotionsregulation kann somit als eine Sammlung von kogn. und verhaltensbasierten Strategien zur Beseitigung, Aufrechterhaltung und Veränderung von emot. Erleben und Ausdruck aufgefasst werden. Damit sind generell alle Prozesse gemeint, welche die spontane Entfaltung von Emotionen beeinflussen im Hinblick darauf, welche Emotionen wir haben, wann wir diese haben und wie wir diese erleben und im Verhalten (z. B. Gestik, Mimik) zum Ausdruck bringen. Die Intensität von sowohl pos. als auch neg. Emotionen kann in jede Richtung beeinflusst werden. In der psychol. Emotionsregulation-Forschung interessiert jedoch meist die Verringerung neg. Emotionen: effektive Emotionsregulation besteht demnach darin, pos. Emotionen aufrechtzuerhalten und neg. Emotionen zu verringern."""

DEFAULT_QUESTIONS = [
    "Im Ärger werde ich manchmal auch lauter.",
    "Wenn es die Situation erfordert, kann ich nach außen hin meine wahren Gefühle verbergen.",
    "Es fällt mir leicht, meine Gefühle bewusst zu verändern.",
    "Wenn ich einmal in schlechter Stimmung bin, kann ich diese immer bewusst verbessern.",
    "Wenn ich will, kann ich mich in eine gute Stimmung bringen.",
    "Selbst starke Erregung und Wut kann ich nach außen besser verbergen als andere.",
    "Wenn ich gereizt und zornig bin, kann ich mich besser beherrschen als andere.",
    "Es fällt mir schwer, meine Gedanken und Emotionen zu kontrollieren, wenn es stressig wird.",
    "Ich habe häufig unkontrollierte Gefühlsausbrüche.",
    "Ich bin ein Biologe."
]

# Initialize session state
if 'questions' not in st.session_state:
    st.session_state.questions = [{'id': str(uuid.uuid4()), 'text': q} for q in DEFAULT_QUESTIONS]

if 'last_added_items' not in st.session_state:
    st.session_state.last_added_items = None

if 'generated_items' not in st.session_state:
    st.session_state.generated_items = None

# Initialize the item generator with Anthropic API
@st.cache_resource
def initialize_item_generator():
    try:
        anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        item_generator = GermanPersonalityItemGenerator(anthropic_client)
        return item_generator
    except Exception as e:
        st.error(f"Fehler bei der Initialisierung des Item-Generators: {str(e)}")
        return None

@st.cache_resource
def load_models():
    try:
        with st.spinner('Lade Modelle... Dies kann einen Moment dauern.'):
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
            bert_model = BertModel.from_pretrained('bert-base-german-cased')
            sbert_model = SentenceTransformer('deutsche-telekom/gbert-large-paraphrase-cosine')
            return bert_tokenizer, bert_model, sbert_model
    except Exception as e:
        st.error(f"Fehler beim Laden der Modelle: {str(e)}")
        return None, None, None

@st.cache_data
def bert_sentence_embedding(sentence, _model, _tokenizer):
    inputs = _tokenizer(
        sentence,
        return_tensors='pt',
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=512
    )
    
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = _model(**inputs)
    
    token_embeddings = outputs.last_hidden_state.squeeze(0)
    attention_mask = attention_mask.squeeze(0)
    
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, 0)
    sum_mask = torch.clamp(mask_expanded.sum(0), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    
    return mean_embeddings.numpy()

def main():
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("AI Item-Entwicklungs-Tool")
    with col2:
        st.image("assets/sapientia-metrics-logo.svg", width=180)

    # Add explanation box
    st.markdown("""
    ### Über dieses Tool
    Die Entwicklung von psychometrischen Fragebögen ist ein **zeitaufwändiger und kostspieliger** Prozess. Sollen neue Fragen (Items) hinzugefügt werden, 
    muss ein Fragebogen typischerweise einer Stichprobe von ProbandInnen vorgelegt werden, damit empirisch erhoben werden kann, wie gut die neuen Items
    das intendierte Konstrukt messen. Nach einer Pilotstudie und der Entfernung der Items, die sich nicht bewähren, muss der vorläufig finale Fragebogen nochmals einer
    größeren Stichprobe vorgelegt werden, um die endgültigen Gütekriterien zu erheben.

    **Dieses Tool unterstützt Sie bei der Entwicklung von Items, indem es die semantische Ähnlichkeit zwischen einer 
    Konstruktdefinition und möglichen Items berechnet.** Ähnlichkeiten zwischen allen einzelnen Items werden ebenfalls in Form einer Heatmap dargestellt. 
    Diese Ähnlichkeiten können als (grobe) Schätzer für die Zusammenhänge zwischen Items und Konstrukt bzw. Items untereinander angesehen werden.
    **Dadurch verkürzt das Tool die ersten Schritte der Testentwicklung bzw. -optimierung enorm.** Neue potenzielle Items können in Sekunden erprobt werden. 
    Es müssen zudem keine empirischen Daten erhoben werden. Die Wahrscheinlichkeit, "auf Anhieb" gut funktionierende, valide Items zu konstruieren, steigt deutlich an.            

    Die Analyse basiert auf zwei verschiedenen Sprachmodellen (BERT und SBERT), die speziell für die deutsche Sprache optimiert wurden.
    **Die Ähnlichkeitswerte reichen von 0 (keine Ähnlichkeit) bis 1 (perfekte Ähnlichkeit).**
    """)

    st.info("Die App wurde inspiriert durch [diesen Blog-Artikel von Damiano D'Urso](https://damianodurso.github.io/Sentence-embeddings-for-Employee-Listening/).", icon="ℹ️")

    # Optional: Add more detailed information in an expandable section
    with st.expander("Detaillierte Informationen"):
        st.markdown("""
        **Funktionsweise:**
        1. Geben Sie eine Konstruktdefinition ein oder nutzen Sie das vorgegebene Beispiel
        2. Fügen Sie potenzielle Items hinzu
        3. Starten Sie die Analyse
        4. Prüfen Sie die Ergebnisse in den verschiedenen Tabs:
            - BERT Ergebnisse
            - SBERT Ergebnisse
            - Item-Ähnlichkeiten (Heatmap)
        """)

    # Load models
    tokenizer, model, sbert_model = load_models()

    if tokenizer is None or model is None or sbert_model is None:
        st.error("Fehler beim Laden der Modelle. Bitte laden Sie die Seite neu, um es erneut zu versuchen.")
        st.stop()

    # Sidebar for instructions
    with st.sidebar:
        st.header("Anleitung")
        st.write("""
        1. Geben Sie die Konstruktdefinition ein
        2. Fügen Sie Items hinzu
        3. Klicken Sie auf 'Analyse starten'
        """)

        st.markdown("---")
        st.markdown(
            """
            <div style='display: flex; align-items: center; gap: 10px;'>
                <a href="https://www.linkedin.com/in/timo-krug/" style='display: flex; align-items: center; color: #0E165A; text-decoration: none; gap: 5px;'>
                    <img src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" 
                    width="18px"/> Timo Krug auf LinkedIn
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Main content area
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Konstruktdefinition")
        construct = st.text_area(
            "Definition eingeben:",
            value=DEFAULT_CONSTRUCT,
            height=200
        )

    # Header for items section
    st.header("Items")
    st.markdown("""
    Wählen Sie eine der folgenden Methoden, um Items hinzuzufügen.
    """)

    # Create tabs for different item input methods
    tab_manual, tab_file, tab_generate = st.tabs([
        "Manuelle Eingabe", 
        "Datei-Upload", 
        "KI-Generierung"
    ])

    # Tab 1: Manual Input
    with tab_manual:
        col3, col4 = st.columns([1, 5])
        with col3:
            if st.button("➕ Item hinzufügen"):
                st.session_state.questions.append({'id': str(uuid.uuid4()), 'text': ''})
                st.rerun()
        
        for i, item in enumerate(st.session_state.questions):
            col_del, col_input = st.columns([1, 7])
            with col_del:
                if st.button("🗑️", key=f"del_{item['id']}"):
                    st.session_state.questions = [q for q in st.session_state.questions if q['id'] != item['id']]
                    st.rerun()
            with col_input:
                new_text = st.text_input(
                    f"Item {i+1}",
                    value=item['text'],
                    key=f"q_{item['id']}"
                )
                # Update the item text if it changed
                if new_text != item['text']:
                    item['text'] = new_text

    # Tab 2: File Upload
    with tab_file:
        uploaded_file = st.file_uploader("Excel/CSV Import", type=['xlsx', 'csv'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                if len(df.columns) > 1:
                    column = st.selectbox("Wählen Sie die Spalte mit den Items:", df.columns)
                else:
                    column = df.columns[0]
                
                if st.button("Items importieren"):
                    new_items = df[column].dropna().tolist()
                    for q in new_items:
                        st.session_state.questions.append({'id': str(uuid.uuid4()), 'text': q})
                    st.success(f"{len(new_items)} Items erfolgreich importiert!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Fehler beim Einlesen der Datei: {str(e)}")

    # Tab 3: AI Generation
    with tab_generate:
        generator = initialize_item_generator()
        if generator is None:
            st.error("Item-Generator konnte nicht initialisiert werden.")
            st.stop()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            n_items = st.slider("Anzahl Items", min_value=5, max_value=30, value=10)
        with col2:
            negative_ratio = st.slider("Anteil negativer Items", min_value=0.0, max_value=0.5, value=0.3)
        with col3:
            work_context = st.checkbox("Arbeitskontext", value=False)

        generate_button = st.button("Items Generieren", type="primary")
        
        if generate_button:
            if not construct:
                st.error("Bitte geben Sie eine Konstrukt-Definition ein.")
                st.stop()
                
            try:
                with st.spinner("Generiere Items..."):
                    generated_items = generator.generate_items(
                        construct_definition=construct,
                        n_items=n_items,
                        work_context=work_context,
                        negative_ratio=negative_ratio
                    )
                    st.session_state.generated_items = generated_items
                    # Remove this line to prevent double display
                    # generator.format_results_for_display(generated_items)

                # Create two columns for the buttons
                col1, col2 = st.columns(2)
                
                # Column 1: Button to use items
                with col1:
                    use_items_button = st.button(
                        "Generierte Items für Analyse verwenden",
                        key="use_items_button",
                        type="primary"
                    )

                # Column 2: Download button
                with col2:
                    if isinstance(generated_items, dict):
                        all_items = generated_items.get('all_items', [])
                        if not all_items:
                            all_items = generated_items.get('positive', []) + generated_items.get('negative', [])
                    elif isinstance(generated_items, list):
                        all_items = generated_items
                    else:
                        all_items = []

                    df = pd.DataFrame({'Items': all_items})
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Items als CSV herunterladen",
                        data=csv,
                        file_name="generierte_items.csv",
                        mime="text/csv",
                        key="download_items"
                    )
                    
            except Exception as e:
                st.error(f"Fehler bei der Item-Generierung: {str(e)}")
                st.write("Details zum Fehler:", str(e))

        # Check if the use items button was clicked
        if 'use_items_button' in st.session_state and st.session_state.use_items_button:
            if 'generated_items' in st.session_state and st.session_state.generated_items:
                generated_items = st.session_state.generated_items
                all_items = generated_items.get('all_items', [])
                if not all_items:
                    all_items = generated_items.get('positive', []) + generated_items.get('negative', [])
                
                if all_items:
                    # Add new items to session state
                    for item_text in all_items:
                        st.session_state.questions.append({
                            'id': str(uuid.uuid4()),
                            'text': item_text
                        })
                    st.success(f"{len(all_items)} Items wurden erfolgreich zur Analyse hinzugefügt!")
                    
                    # Clear the generated items to prevent duplicate additions
                    st.session_state.generated_items = None
                    st.rerun()
                else:
                    st.error("Keine Items zum Hinzufügen gefunden.")
            else:
                st.error("Bitte generieren Sie zuerst neue Items.")

        # Display generated items in expander if they exist
        if 'generated_items' in st.session_state and st.session_state.generated_items:
            with st.expander("🔍 Generierte Items anzeigen", expanded=True):
                generator.format_results_for_display(st.session_state.generated_items)

    # Display current items
    st.divider()
    st.subheader("Aktuelle Items für die Analyse")
    if st.session_state.questions:
        for i, item in enumerate(st.session_state.questions, 1):
            st.write(f"{i}. {item['text']}")
    else:
        st.info("Noch keine Items für die Analyse ausgewählt.")

    # Analysis button and results
    if st.button("🔍 Analyse starten", type="primary"):
        if not st.session_state.questions:
            st.error("Bitte fügen Sie mindestens ein Item hinzu.")
        else:
            with st.spinner("Analysiere Items..."):
                # Get current questions from session state
                questions = [item['text'] for item in st.session_state.questions if item['text'].strip()]
                
                # Generate embeddings for construct
                construct_embedding_bert = bert_sentence_embedding(construct, model, tokenizer)
                construct_embedding_sbert = sbert_model.encode(construct, normalize_embeddings=True)
                
                # Generate embeddings for questions
                bert_embeddings = [bert_sentence_embedding(q, model, tokenizer) for q in questions]
                sbert_embeddings = sbert_model.encode(questions, normalize_embeddings=True)
                
                # Calculate similarities
                similarities_bert = [abs(cosine_similarity([construct_embedding_bert], [embedding])[0][0]) 
                                   for embedding in bert_embeddings]
                similarities_sbert = [abs(cosine_similarity([construct_embedding_sbert], [embedding])[0][0])
                                    for embedding in sbert_embeddings]
                
                # Create tabs for results
                tab1, tab2, tab3, tab4 = st.tabs([
                    "BERT Ergebnisse", 
                    "SBERT Ergebnisse", 
                    "Item-Ähnlichkeiten", 
                    "Pseudo-Faktorenanalyse"
                ])
                
                with tab1:
                    st.subheader("Top 5 Items (BERT)")
                    results_bert = list(zip(questions, similarities_bert))
                    results_bert.sort(key=lambda x: x[1], reverse=True)
                    
                    for i, (question, similarity) in enumerate(results_bert[:5], 1):
                        st.markdown(f"**{i}. Item:** {question}")
                        st.progress(float(similarity))
                        st.markdown(f"Ähnlichkeit: {similarity:.4f}")
                        st.divider()
                
                with tab2:
                    st.subheader("Top 5 Items (SBERT)")
                    results_sbert = list(zip(questions, similarities_sbert))
                    results_sbert.sort(key=lambda x: x[1], reverse=True)
                    
                    for i, (question, similarity) in enumerate(results_sbert[:5], 1):
                        st.markdown(f"**{i}. Item:** {question}")
                        st.progress(float(similarity))
                        st.markdown(f"Ähnlichkeit: {similarity:.4f}")
                        st.divider()

                with tab3:
                    st.subheader("Paarweise Ähnlichkeiten zwischen Items")
                    
                    # Calculate pairwise similarities
                    n_items = len(questions)
                    sbert_similarity_matrix = np.zeros((n_items, n_items))
                    
                    for i in range(n_items):
                        for j in range(n_items):
                            similarity = abs(cosine_similarity([sbert_embeddings[i]], 
                                                            [sbert_embeddings[j]])[0][0])
                            sbert_similarity_matrix[i, j] = similarity
                    
                    # Create heatmap
                    fig = px.imshow(
                        sbert_similarity_matrix,
                        labels=dict(x="Items", y="Items", color="Ähnlichkeit"),
                        x=questions,
                        y=questions,
                        color_continuous_scale=[[0, "white"], [1, "#0E165A"]],
                        aspect="auto"
                    )

                    fig.update_layout(
                        width=1000,
                        height=1000,
                        title="Heatmap der Item-Ähnlichkeiten",
                        xaxis_tickangle=-45
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                with tab4:
                    st.subheader("Exploratorische Faktorenanalyse")
                    
                    # Use SBERT similarity matrix as correlation matrix
                    similarity_matrix = np.zeros((len(questions), len(questions)))
                    for i in range(len(questions)):
                        for j in range(len(questions)):
                            similarity = abs(cosine_similarity([sbert_embeddings[i]], 
                                                            [sbert_embeddings[j]])[0][0])
                            similarity_matrix[i, j] = similarity
                    
                    # Perform EFA
                    fa = FactorAnalyzer(rotation="varimax", n_factors=1, is_corr_matrix=True)
                    fa.fit(similarity_matrix)
                    
                    # Get factor loadings
                    loadings = pd.DataFrame(
                        fa.loadings_,
                        columns=['Factor 1'],
                        index=questions
                    )
                    
                    # Display results
                    st.write("Faktorladungen:")
                    formatted_loadings = loadings.copy()
                    formatted_loadings['Factor 1'] = formatted_loadings['Factor 1'].round(3)
                    formatted_loadings = formatted_loadings.sort_values('Factor 1', ascending=False)
                    st.dataframe(
                        formatted_loadings, 
                        column_config={
                            "Factor 1": st.column_config.NumberColumn(
                                "Faktorladung",
                                format="%.3f"
                            )
                        },
                        height=400
                    )
                    
                    # Scree plot
                    st.subheader("Scree Plot")
                    eigenvalues = np.linalg.eigvals(similarity_matrix)
                    eigenvalues = np.sort(eigenvalues)[::-1]
                    variance = eigenvalues / len(eigenvalues)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(eigenvalues) + 1)),
                        y=eigenvalues,
                        mode='lines+markers',
                        name='Eigenwerte',
                        line=dict(color='#0E165A'),
                        marker=dict(size=10)
                    ))

                    fig.add_hline(
                        y=1, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text="Kaiser-Kriterium (Eigenwert = 1)", 
                        annotation_position="bottom right"
                    )

                    fig.update_layout(
                        title='Scree Plot zur Bestimmung der Faktorenanzahl',
                        xaxis_title='Faktor Nummer',
                        yaxis_title='Eigenwert',
                        template='plotly_white',
                        showlegend=True,
                        width=800,
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Interpretation
                    st.markdown("""
                    ### Interpretation des Scree Plots:
                    - Faktoren mit Eigenwerten > 1 (über der roten Linie) sollten nach dem Kaiser-Kriterium beibehalten werden
                    - Der "Ellbogen" (starker Knick im Verlauf) deutet auf die optimale Faktorenanzahl hin
                    """)

                    # Eigenvalues table
                    eigenvalue_df = pd.DataFrame({
                        'Faktor': range(1, len(eigenvalues) + 1),
                        'Eigenwert': eigenvalues.round(3),
                        'Erklärte Varianz (%)': (variance * 100).round(2),
                        'Kumulierte Varianz (%)': (np.cumsum(variance) * 100).round(2)
                    })

                    st.write("Eigenwerte und erklärte Varianz:")
                    st.dataframe(eigenvalue_df)

if __name__ == "__main__":
    main()