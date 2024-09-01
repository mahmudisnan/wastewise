import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import io

# Set Streamlit page configuration
st.set_page_config(page_title="WasteWiseChatbot", layout="centered")

# Load and display the logo in the sidebar
logo = "logo.jpg"  # Replace with your logo file name
st.sidebar.image(logo, use_column_width=True)

# Sidebar for API key input and temperature slider
st.sidebar.title("Configuration")




api_key_input = st.sidebar.text_input("Qwen API Key", type="password", key="api_key")
temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# Store the API key in the session state
if api_key_input:
    st.session_state["openai_api_key"] = api_key_input

# Use the API key from session state
api_key = st.session_state.get("openai_api_key", None)

if not api_key:
    st.sidebar.error("Please enter your Qwen API key to continue.")
    st.stop()

# Instantiate the OpenAI client
client = OpenAI(api_key=api_key)

# Load CSV data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

data = load_data('datasampah1.csv')

# Display data if necessary
if st.checkbox('Show CSV Data'):
    st.write(data)

# Initialize session state to store responses and plots
if "responses" not in st.session_state:
    st.session_state["responses"] = []

if "plots" not in st.session_state:
    st.session_state["plots"] = []

# Function to generate responses using OpenAI's API
def generate_response(prompt, temperature=0.7):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "I'm sorry, I couldn't generate a response."

# Function to parse the prompt and generate a graph
def parse_and_generate_graph(prompt):
    # Lowercase and strip the prompt for consistency
    prompt = prompt.lower().strip()
    
    # Example: "make a line plot of Column1 vs Column2"
    if "line plot" in prompt:
        graph_type = "Line Plot"
    elif "bar plot" in prompt:
        graph_type = "Bar Plot"
    elif "scatter plot" in prompt:
        graph_type = "Scatter Plot"
    else:
        st.error("Could not understand the type of graph you want. Please specify 'line plot', 'bar plot', or 'scatter plot'.")
        return
    
    # Extract columns from the prompt
    words = prompt.split()
    try:
        x_axis = words[words.index("dan") - 1]
        y_axis = words[words.index("dan") + 1]
    except (ValueError, IndexError):
        st.error("Could not understand the columns for the graph. Please use the format 'Column1 dan Column2'.")
        return
    
    if x_axis not in data.columns or y_axis not in data.columns:
        st.error("The specified columns do not exist in the data.")
        return
    
    # Generate the graph
    st.subheader(f"{graph_type}  {y_axis} dan {x_axis}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    if graph_type == "Line Plot":
        sns.lineplot(x=data[x_axis], y=data[y_axis], ax=ax)
    elif graph_type == "Bar Plot":
        sns.barplot(x=data[x_axis], y=data[y_axis], ax=ax)
    elif graph_type == "Scatter Plot":
        sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
    
    ax.set_title(f"{graph_type} {y_axis} dan {x_axis}")
    
    # Save the figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    
    # Store the plot in the session state
    st.session_state["plots"].append(buf)
    st.pyplot(fig)
    plt.close(fig)


# User input
question = st.text_input("Ask a question or make a request (e.g., 'buat line plot Column1 dan Column2'):")

if question:
    if "buat" in question.lower() and "plot" in question.lower():
        parse_and_generate_graph(question)
    else:
        # Formulate the prompt for the AI
        prompt = f"Based on the data: \n\n{data.to_string(index=False)}\n\nQ: {question}\nA:"
        answer = generate_response(prompt, temperature)
        
        # Store the new response in session state
        st.session_state["responses"].append({"question": question, "answer": answer})

# Display all previous responses and plots in reverse order (latest first)
for i, entry in enumerate(reversed(st.session_state["responses"])):
    st.write(f"**Q:** {entry['question']}")
    st.write(f"**A:** {entry['answer']}")
    if i < len(st.session_state["plots"]):
        st.image(st.session_state["plots"][-(i+1)], use_column_width=True)
    st.write("---")  # Separator line
