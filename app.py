import streamlit as st
import joblib
from prediction_pipeline import *
from PIL import Image


st.set_page_config(page_title="Game of Thrones",
                   page_icon="👑", layout="wide")

# creating a side bar 
sidebar_image = Image.open('GOT.jpg')
st.sidebar.info("Created By : Priyesh Dave")
# Adding an image to the side bar 
st.sidebar.image(sidebar_image, width=None)
st.sidebar.subheader("Contact Information : ")
col1, mid, col2 = st.columns([1,1,20])
with col1:
	st.sidebar.subheader("LinkedIn : ")
with col2:
	st.sidebar.markdown("[![Linkedin](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTLsu_X_ZxDhuVzjTHvk4eZOmUDklreUExhlw&usqp=CAU)](https://www.linkedin.com/in/priyeshdave21/)")

col3, mid, col4 = st.columns([1,1,20])
with col3:
	st.sidebar.subheader("Github : ")
with col4:
	st.sidebar.markdown("[![Github](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJGtP-Pq0P67Ptyv3tB7Zn2ZYPIT-lPGI7AA&usqp=CAU)](https://github.com/PriyeshDave)")
data = read_file()

#creating option list for dropdown menu
st.markdown("<h1 style='text-align: center;'>Game of Thrones 👑</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Text Generation Application 📃</h1>", unsafe_allow_html=True)
image_cyber_bully = Image.open('Text Generation.jpg')
st.image(image_cyber_bully)

st.download_button(label="Click here to download to text file", data=data, file_name='Cleaned-Text.txt', mime='text/csv')
def main():
    with st.form('prediction_form'):


        st.subheader("Enter the sentence below.")
        comment = st.text_input(label='Enter text here...')
        n_words = st.slider("Pickup number of words to be predicted: ", 1, 50, value=1, format="%d")
        submit = st.form_submit_button("Predict")


    if submit:
       pred = get_predictions(comment, n_words)
    
       st.write(pred)

if __name__ == '__main__':
    main()