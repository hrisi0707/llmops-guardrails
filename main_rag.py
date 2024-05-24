import os
import json
import streamlit as st
from langsmith import Client
from streamlit_feedback import streamlit_feedback
from rag_chain import get_expression_chain
from langchain_core.tracers.context import collect_runs
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from scripts.guards import prompt_scanner,response_scanner
from scripts import hallucination
import pandas as pd

# Index Name
index_name = 'serverless-index'

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGSMITH"]['LANGSMITH_API_KEY']
#os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI"]["OPENAI_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE"]['PINECONE_API_KEY']
os.environ['LANGCHAIN_PROJECT'] = st.secrets["LANGSMITH"]["LANGCHAIN_PROJECT"]
os.environ['LANGCHAIN_PROJECT'] = st.secrets["LANGSMITH"]["LANGCHAIN_PROJECT"]
os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AzureOpenAI"]['AZURE_OPENAI_API_KEY']
os.environ["AZURE_OPENAI_EMBEDDING_MODEL_NAME"] = st.secrets["AzureOpenAI"]['AZURE_OPENAI_EMBEDDING_MODEL_NAME']
os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AzureOpenAI"]['AZURE_OPENAI_ENDPOINT']

client = Client()
#embeddings = OpenAIEmbeddings()
embeddings = AzureOpenAIEmbeddings(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],model=os.environ["AZURE_OPENAI_EMBEDDING_MODEL_NAME"])
# loading filenames to show in streamlit app
with open("mappings.json", 'r') as json_file:
    mappings = json.load(json_file)

@st.cache_resource(show_spinner=False)
def load_data():
    index = None
    try:
        index = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        print("Index:", index)
    except Exception as e:
        print(f"Could not load index: {e}")        
    return index

# Initializing index
if "index" not in st.session_state.keys():
	st.session_state.index = load_data()

st.set_page_config(
    page_title="Capturing User Feedback",
    page_icon="🦜️️🛠️",
)

st.subheader("🦜🛠️ Fractal Finance Bot")

# Metadata from user
with st.sidebar:
    year = st.selectbox("Select Year",list(mappings.keys()))
    quarter = st.selectbox("Select Quarter",list(mappings[year].keys()))
    file_name = st.selectbox("Select file", mappings[year][quarter])
	
metadata={"filename":file_name+".pdf","year":year,"quarter":quarter}

# Initializing session metadata
if "metadata" not in st.session_state.keys():
    st.session_state.metadata = metadata

# Initializing Query Engine
if "retriever" not in st.session_state.keys():
    st.session_state.retriever = st.session_state.index.as_retriever(search_kwargs={"filter": st.session_state.metadata, "k": 4})
    st.session_state.chain = get_expression_chain(retriever=st.session_state.retriever)
    hallucination.init(rag_pipeline=st.session_state.chain)

# Updating filters and chat engine if metadata is updated and updating session metadata also
if st.session_state.metadata != metadata:
    st.session_state.metadata = metadata
    st.session_state.retriever = st.session_state.index.as_retriever(search_kwargs={"filter": st.session_state.metadata, "k": 4})
    st.session_state.chain = get_expression_chain(retriever=st.session_state.retriever)
    hallucination.init(rag_pipeline=st.session_state.chain)

st.sidebar.markdown("## Feedback Scale")
feedback_option = (
    "thumbs" if st.sidebar.toggle(label="`Faces` ⇄ `Thumbs`", value=False) else "faces"
)

# Initialize session state
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

for msg in st.session_state.messages:
    print("MESSAGE:", msg)
    avatar = "🦜" if msg.get("type") == "ai" else None
    with st.chat_message(msg.get("type"), avatar=avatar):
        st.markdown(msg.get("content"))


if prompt := st.chat_input(placeholder="Ask me a question!"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"type": "user", "content": prompt})
    with st.spinner(text="Thinking ..."):
        with st.chat_message("assistant", avatar="🦜"):
            message_placeholder = st.empty()
            full_response = ""
            # Define the basic input structure for the chains
            # Scanning the prompt
            scanned_prompt = prompt_scanner(prompt=prompt)
            st.dataframe(scanned_prompt,use_container_width=True,hide_index=True)
            
            input_dict = {"query_str": prompt}
            with collect_runs() as cb:
                full_response = st.session_state.chain.invoke(prompt)
                st.session_state.messages.append({
                    "type": "ai",
                    "content": full_response.get("answer")
                })
                st.session_state.run_id = cb.traced_runs[0].id

            # Scanning the response
            scanned_response = response_scanner(response=full_response.get("answer"))
            
            # Checking for hallucination
            hallucination_result = hallucination.consistency_check(prompt=prompt,response=full_response.get("answer"))

            other_features = [
                {'Metrics': 'Hallucination Score', 'Value': str(round(hallucination_result['final_score'] * 100, 2)) + '%'},
                {'Metrics': 'Interpretability', 'Value': 'See the context below'}]
            
            other_features_df = pd.DataFrame(other_features)

            # Making combined response
            combined_response = pd.concat([scanned_response, other_features_df], ignore_index=True)
            
            # Displaying the combined response in the frontend
            st.dataframe(combined_response,use_container_width=True,hide_index=True)

            meta_data_for_user_context = {}
            for index, item in enumerate(full_response.get('context_str')):
                    meta_data_for_user_context[f'Retrived_Chunk_{index+1}'] = item.metadata
                    meta_data_for_user_context[f'Retrived_Chunk_{index+1}'].update({"source": item.page_content})
                    
                    # showing metadata for user context
                    st.dataframe(pd.Series(meta_data_for_user_context[f'Retrived_Chunk_{index+1}']), use_container_width=True, column_config={1: "Retrived Chunk"})

            st.write(scanned_response.iloc[1]["Value"].replace("$", "\$"))

            message_placeholder.markdown(full_response.get("answer"))

if st.session_state.get("run_id"):
    run_id = st.session_state.run_id
    feedback = streamlit_feedback(
        feedback_type=feedback_option,
        optional_text_label="[Optional] Please provide an explanation",
        key=f"feedback_{run_id}",
    )

    # Define score mappings for both "thumbs" and "faces" feedback systems
    score_mappings = {
        "thumbs": {"👍": 1, "👎": 0},
        "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
    }

    # Get the score mapping based on the selected feedback option
    scores = score_mappings[feedback_option]

    if feedback:
        # Get the score from the selected feedback option's score mapping
        score = scores.get(feedback["score"])

        if score is not None:
            # Formulate feedback type string incorporating the feedback option
            # and score value
            feedback_type_str = f"{feedback_option} {feedback['score']}"

            # Record the feedback with the formulated feedback type string
            # and optional comment
            feedback_record = client.create_feedback(
                run_id,
                feedback_type_str,
                score=score,
                comment=feedback.get("text"),
            )
            st.session_state.feedback = {
                "feedback_id": str(feedback_record.id),
                "score": score,
            }
        else:
            st.warning("Invalid feedback score.")