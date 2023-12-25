import streamlit as st
from melodygenerator_gru import MelodyGenerator_g
from melodygenerator_lstm import MelodyGenerator_l
from preprocess import SEQUENCE_LENGTH
from util import set_background

# Set the title
st.title("Welcome to Music Generation Using AI")

# Add a text input area for the seed
seed1 = st.text_input("Seed", "")

# Create buttons to generate melodies using LSTM and GRU
if st.button("Generate LSTM Melody"):
    # set_background('pix/lstm.jpg')
    # Initialize the MelodyGenerator for LSTM
    mgl = MelodyGenerator_l()

    # Generate melody using LSTM
    melody = mgl.generate_melody(seed1, num_steps=500, max_sequence_length=SEQUENCE_LENGTH, temperature=0.2)
    
    # Save the generated melody
    mgl.save_melody(melody, file_name="lstm_melody.mid")
    st.write("LSTM Melody generated and saved to the folder!")

    
    
if st.button("Generate GRU Melody"):
    # Initialize the MelodyGenerator for GRU
    # set_background('pix/gru.jpg')
    mg = MelodyGenerator_g()

    # Generate melody using GRU
    melody = mg.generate_melody(seed1, num_steps=500, max_sequence_length=SEQUENCE_LENGTH, temperature=0.2)
    
    # Save the generated melody
    mg.save_melody(melody, file_name="gru_melody.mid")
    st.write("GRU Melody generated and saved to the folder!")
 
