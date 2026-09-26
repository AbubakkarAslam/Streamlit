import streamlit as st
import numpy as np
import pandas as pd

# Adding the title 
st.title("My first testing app for learning the Streamlit")

# add the simple text
st.write("helo")


# add the slider
number = st.slider("pick the number", 0,100)

# print the number in the text form 
st.write(f"you selected the number {number}")

# add the slider defaut the 10 
number = st.slider("pick the number", 0,100,10)

# print the number in the text form 
st.write(f"you selected the number {number}")

# adding the button with conditions
if st.button("Click"):
    st.write("Hi, Hello there")
    st.write("you clicked on the button")
else:
    st.write("GoodBy")

# add radio button with options   

genre = st.radio(
    "What's your favourite movie genre",
    ('Comedy', 'Drama','Documentary')
)
# print the genre 
st.write(f'You selected the genre : {genre}')

# add the drop down list option
option = st.selectbox(
    'How would you like to be contacted',
    ('Email','Home Phone', 'Mobile Phone')
)

# print the selected option
st.write(f'You selected the: {option}')


# add the drop down list on the left side bar 
option = st.sidebar.selectbox(
    'How would you ike to connected',
    ('Email', 'Home phone', 'Mobile phone')
)
# add your text data from the user
st.text_input('Enter your Whatsapp number')


# add your text data from the user on the left side bar
st.sidebar.text_input('Enter your Whatsapp number')

# upload the file 
upload_file = st.sidebar.file_uploader("Choose the CSV file", type=['csv, pdf'])

# create a line plot
data = pd.DataFrame({
    'first column': list(range(1,11)),
    'second column': np.arange(number, number + 10)

})
st.line_chart(data)