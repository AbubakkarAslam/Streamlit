import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# add title 
st.write("Data Analysis Application")
st.subheader("This is a simple data analysis application created by @abubakar")

# make dunction in which a user will choose the dataset from the dropdown button
dataset_option = ['iris','titanic', 'tips']

# for drop down the list 
selected_dataset = st.selectbox("Select the dataset", dataset_option)

if selected_dataset == 'iris':
    df = sns.load_dataset('iris')
elif selected_dataset == 'tips':
    df = sns.load_dataset("tips")
elif selected_dataset == 'titanic':
    df = sns.load_dataset("titanic")


uploaded_file = st.file_uploader("Upload the dataset", type = ['csv','xlsx'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

# Display the dataset 
# st.write(df.head()) or 
st.write(df)

# Display the number of cloumns and number of rows
st.write(f"The total number of columns {df.shape[0]} and the number of rows {df.shape[1]} in the **{selected_dataset}**")


# display the columns name and their dtype in the selected dataset 
st.subheader(selected_dataset)
st.write(df.dtypes)


# print the total null values in the selected dataset


# st.write(df.isnull().sum().sort_values(ascending=False))
# or with the condition

if df.isnull().sum().sum() > 0:
    st.write("Null Values : ", df.isnull().sum().sort_values(ascending=False))
else:
    st.write("Not Null Values")

# Display the whole summary of the dataset 
st.write('The summary of the whole dataset ',df.describe())


# Select the column and row X_axis and y_axis  and display the plot based on these axis 
# x_axis = st.selectbox("Select the columns ", df.columns)
# y_axis = st.selectbox("Select the rows", df.columns)
# select_plot = st.selectbox("Slect the plot ", ['line', 'scatter', 'bar', ' hist', 'box'])
# if select_plot == 'line':
#     st.line_chart(df[[x_axis, y_axis]])
# elif select_plot == 'scatter':
#     st.scatter_chart(df[[x_axis, y_axis]])
# elif select_plot == 'bar':
#     st.bar_chart(df[[x_axis, y_axis]])
# elif select_plot == 'hist':
#     df[x_axis].plot(kind='hist')
#     st.pyplot()
# elif select_plot == 'box':
#     df[x_axis, y_axis].plot(kind='box')
#     st.pyplot()

# creat the pair plot 
st.subheader("Pair plot")

# select the column for hue 
select_hue_column = st.selectbox("Select the column for and make the color ", df.columns)
st.pyplot(sns.pairplot(df, hue = select_hue_column))
