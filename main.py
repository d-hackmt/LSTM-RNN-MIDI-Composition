import streamlit as st
st.set_page_config(layout="wide")
from melodygenerator_gru import MelodyGenerator_g
from melodygenerator_lstm import MelodyGenerator_l
from preprocess import SEQUENCE_LENGTH
from util import set_background
import random
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

# Set the title
# st.title("Music Generation & LLM Chatbot")

st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.write("")

st.markdown('<h1 class="centered-title">MIDI Generation & LLM Chatbot</h1>', unsafe_allow_html=True)

st.write("")
st.write("")
st.write("")
st.write("")
st.markdown('<div style="text-align:center; font-size: xx-large;">'
            '<a href="http://www.esac-data.org/" style="margin: 0px 20px;">Dataset</a>'
            '<a href="https://www.canva.com/design/DAFvvv3p0js/ynlXWPtar4-FljmPFR-aKw/edit?utm_content=DAFvvv3p0js&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton" style="margin: 0px 20px;">Preprocessing</a>'
            '<a href="https://www.geeksforgeeks.org/understanding-of-lstm-networks/" style="margin: 0px 20px;">LSTM</a>'
            '<a href="https://www.geeksforgeeks.org/gated-recurrent-unit-networks/" style="margin: 0px 20px;">GRU</a>'
            '<a href="https://www.canva.com/design/DAFzIAUYMf8/D9gn80Ii8c8hnGE4gViiaQ/edit?utm_content=DAFzIAUYMf8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton" style="margin: 0px 20px;">METRICS</a>'
            '</div>',
            unsafe_allow_html=True)
st.write("")
st.write("")
st.write("")
# Define the background image or set_background function if required

set_background('pix/3.png')

col1, col2, col3 = st.columns(3)

# Left column for Chatbot
with col1:
    st.header("LLM Chatbot 🤖")
    query = st.text_area('ASK ME ANYTHING', height=70)
    submit = st.button("Ask")

    if submit:
        def getLLMResponse(query):
            llm = CTransformers(model="C:/Users/djadh/PycharmProjects/whisperProject/mistral-7b-instruct-v0.1.Q2_K.gguf",
                                model_type='llama',
                                config={'max_new_tokens': 256,
                                        'temperature': 0.01})

            template = """
            {query}
            """

            prompt = PromptTemplate(
                input_variables=["query"],
                template=template,
            )

            response = llm(prompt.format(query=query))
            return response

        response = getLLMResponse(query)
        st.subheader("LLM Response:")
        st.write(response)

# Right column for Music Generation
with col2:
    st.header("MIDI Generation 🎶")
    # Pre-defined list of seeds
    predefined_seeds = [
        "63 65 _ 61 72 _ _ 60 57 57 65 53 _ _ 75 _ _",
        "54 _ 55 58 _ 57 _ 55 _ 54 _ 55 _ 58 _ 57",
        "60 61 _ 63 65 _ _ 57 55 53 52 50 _ _ 75 _ _",
        # Add more predefined seeds here...
    ]

    # Add a button to generate a random seed from predefined seeds
    if st.button("Generate Random SEQUENCE 🎻"):
        random_seed = random.choice(predefined_seeds)
        st.write(f" SEQUENCE Generated : {random_seed} ")

    # Add a text input area for the seed
    seed1 = st.text_input("GIVE SEQUENCE AS INPUT TO PREDICT THE NEXT SEQUENCE ", "")

    # Create buttons to generate melodies using LSTM and GRU
    if st.button("Generate LSTM Melody"):
        mgl = MelodyGenerator_l()
        melody = mgl.generate_melody(seed1, num_steps=500, max_sequence_length=SEQUENCE_LENGTH, temperature=0.2)
        mgl.save_melody(melody, file_name="lstm_melody.mid")
        st.write("LSTM Melody generated and saved to the folder 🎹")


    if st.button("Generate GRU Melody"):
        mg = MelodyGenerator_g()
        melody = mg.generate_melody(seed1, num_steps=500, max_sequence_length=SEQUENCE_LENGTH, temperature=0.2)
        mg.save_melody(melody, file_name="gru_melody.mid")
        st.write("GRU Melody generated and saved to the folder 🎹")


with col3:
    st.header("Future Work 🔮")

    if st.button("Generate VOCALS   🎤"):
        # Functionality for generating vocals
        st.write("Generating VOCALS...")

    if st.button("Generate Ambience   🎷"):
        # Functionality for generating ambience
        st.write("Generating Ambience...")

    if st.button("Generate Drums   🥁"):
        # Functionality for generating drums
        st.write("Generating Drums...")

    if st.button("Generate Hats   🎚️"):
        # Functionality for generating hats
        st.write("Generating Hats...")