import gradio as gr
import pickle
import base64

# ===============================
# LOAD MODEL & VECTORIZER
# ===============================
with open("dt_disaster_text_data_project_best.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer_disaster.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_text(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return f"Predicted Category: {pred}"

# ===============================
# LOAD BACKGROUND IMAGE (HF SAFE)
# ===============================
def load_bg_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()

bg_image = load_bg_image(
    "stock-vector-a-homeless-man-holding-a-need-help-sign-sitting-down-2522562231.jpg"
)

# ===============================
# CUSTOM CSS (WORKING)
# ===============================
custom_css = f"""
html, body {{
    height: 100%;
    margin: 0;
    overflow: hidden;
}}
body {{
    background-image: url("data:image/jpg;base64,{bg_image}");
    background-repeat: no-repeat;
    background-position: center;
    background-size: cover;
    background-color: #000;
}}
.gradio-container {{
    background: rgba(255, 255, 255, 0.94);
    max-width: 460px;
    margin: auto;
    margin-top: 90px;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.4);
}}
footer {{
    display: none !important;
}}
"""

# ===============================
# GRADIO UI
# ===============================
with gr.Blocks(css=custom_css) as demo:

    gr.Markdown("""
    ## 🚨 Disaster Text Classification  
    **Decision Tree + TF-IDF Model**
    """)

    inp = gr.Textbox(
        label="Input Text",
        placeholder="Enter disaster-related message..."
    )

    out = gr.Textbox(label="Prediction")

    btn = gr.Button("🔍 Predict")
    btn.click(predict_text, inp, out)

demo.launch()

