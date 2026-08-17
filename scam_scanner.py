import streamlit as st
import pandas as pd
import emoji

# Configuración de página
st.set_page_config(
    page_title="Scam Detector",
    layout="centered"
)

# Header
st.title(emoji.emojize(":iphone: Smishing/SPAM detector"))
st.markdown("Analyze whether a text is considered as **spam**, **smishing** or **normal**.")
st.divider()

# Input
string = st.text_area(
    "Write or paste the message to analyze",
    height=120,
    placeholder="Ej: URGENT!!! You have won $500,000..."
)

col1, col2 = st.columns([1, 4])
with col1:
    but = st.button("Analyze", type="primary", icon="🔍", use_container_width=True)

# Resultado
if but and len(string) >= 5:
    with st.spinner("Analizing message..."):
        import pandas as pd
        import re
        from datetime import datetime

        # Compile regular expression patterns
        # Pattern for Email
        email_pattern = re.compile(
            r"([a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*(@|\sat\s)"
            r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(\.|\sdot\s))+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)",
            re.IGNORECASE
        )

        # Pattern for UEL
        url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )

        # Pattern for Phone Number
        phone_pattern = re.compile(r"(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")

        # Detection Functions
        def email_check(text):
            return 1.0 if email_pattern.search(str(text)) else 0.0

        def url_check(text):
            return 1.0 if url_pattern.search(str(text)) else 0.0

        def phone_check(text):
            return 1.0 if phone_pattern.search(str(text)) else 0.0


        df_input = pd.DataFrame()
        df_input["prompt"] = [string]
        df_input["URL"] = [url_check(string)]
        df_input["EMAIL"] = [email_check(string)]
        df_input["PHONE"] = [phone_check(string)]
        
        from transformers import AutoModel, AutoTokenizer
        import joblib
        import torch
        import numpy as np
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        backbone = AutoModel.from_pretrained("dsanchezg05/modernbert-finetuned-backbone")
        tokenizer = AutoTokenizer.from_pretrained("dsanchezg05/modernbert-finetuned-backbone")
        backbone.to(device)
        backbone.eval()
        all_embeddings = []
        max_length = 169
        inputs = tokenizer(
                        df_input["prompt"][0],
                        truncation=True,
                        max_length=max_length,
                        padding=True,           # padding dinámico solo dentro de este batch
                        return_tensors="pt"
                    ).to(device)

        outputs = backbone(**inputs)
        cls_embed = outputs.last_hidden_state[:, 0, :]  # (1, 768)

        df_input["cls_embedding"] = [cls_embed.cpu().tolist()[0]]


        #scale embedding
        loaded_scaler = joblib.load('cls_scaler.pkl')
        df_scaled = df_input.copy()
        cls_embedding = np.stack(df_input['cls_embedding'].to_numpy())
        df_scaled["cls_embedding"] = list(loaded_scaler.transform(cls_embedding))

        #obtain input
        v_concat = np.concatenate([torch.tensor(df_scaled["cls_embedding"]), torch.tensor(np.array(df_scaled["URL"]).reshape(-1, 1)), torch.tensor(np.array(df_scaled["EMAIL"]).reshape(-1, 1)), torch.tensor(np.array(df_scaled["PHONE"]).reshape(-1, 1))],axis=1)
        
        #Model MLP
        # destacar que en moderBERT, cada token posee un embedding de longitud 768
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader

        class MultiClassClassifier(nn.Module):
            def __init__(self, cls_dim = 768, proj_dim = 96, remaining_cols = 3, hidden_dim = 40):
                super().__init__()
                self.proj = nn.Sequential(nn.Linear(cls_dim, proj_dim),
                                        nn.BatchNorm1d(proj_dim),
                                        nn.ReLU())
                                        #   nn.Dropout(dropout))
                # self.motif_mlp = nn.Sequential(nn.Linear(motif_dim, motif_dim),
                #                                nn.BatchNorm1d(motif_dim),
                #                                nn.ReLU(),
                #                                nn.Dropout(dropout))
                self.classifier = nn.Sequential(nn.Linear(proj_dim + remaining_cols, hidden_dim),
                                                nn.BatchNorm1d(hidden_dim),
                                                nn.ReLU(),
                                                # nn.Dropout(dropout),
                                                nn.Linear(hidden_dim, 3)) #! Multi-class logit
            def forward(self, cls_embed, cols_data):
                seq_proj = self.proj(cls_embed)
                # motif_proj = self.motif_mlp(motif_embed)
                x = torch.cat([seq_proj, cols_data], dim=1) #! Concat embeddings
                logits = self.classifier(x)
                return logits
            
            
        #making prediction
        id2label = {0:"NORMAL", 1:"SPAM", 2:"SMISHING"}

        best_model = MultiClassClassifier(cls_dim = 768, proj_dim = 96, remaining_cols = 3, hidden_dim = 40)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        best_model.load_state_dict(torch.load("./dataset_metrics/best_MLP_model.pt", map_location = device))
        best_model.to(device)
        best_model.eval()

        concat_tensor = torch.tensor(v_concat)
        concat_tensor = concat_tensor.to(torch.float32)
        with torch.no_grad():
            logits = best_model(concat_tensor[:,:-3].to(device), concat_tensor[:,-3:].to(device))
            all_probs = torch.sigmoid(logits)  # sigmoid
            cpu_probs = all_probs[0].cpu().numpy()
            y_pred = np.argmax(all_probs.cpu().numpy(), axis=1) #predicted
        # print(string)
        # print(y_pred)
        # print(id2label[y_pred[0]])
        # resultado, probs = predecir(message)
        resultado =  id2label[y_pred[0]]# placeholder

    st.divider()

    # Mostrar resultado con color según categoría
    color_map = {"NORMAL": "green", "SPAM": "orange", "SMISHING": "red"}
    icon_map = {"NORMAL": "✅", "SPAM": "⚠️", "SMISHING": "🚨"}

    st.markdown(f"### {icon_map[resultado]} Result: :{color_map[resultado]}[{resultado}]")
    st.progress(float(cpu_probs[0]), text=f"NORMAL confidence: {float(cpu_probs[0])*100:.1f}%")
    st.progress(float(cpu_probs[1]), text=f"SPAM confidence: {float(cpu_probs[1])*100:.1f}%")
    st.progress(float(cpu_probs[2]), text=f"SMISHING confidence: {float(cpu_probs[2])*100:.1f}%")

    # Features detectadas (opcional, si tienes esos datos)
    st.subheader("Detected signals")
    c1, c2, c3 = st.columns(3)
    c1.metric("URL", "Yes" if url_check(string) else "No")
    c2.metric("Email", "Yes" if email_check(string) else "No")
    c3.metric("Phone", "Yes" if phone_check(string) else "No")

elif (but and not string) or len(string) < 5:
    st.warning("Please, submit a message to analyze.")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About the model")
    st.markdown("""
    This classifier leverages CLS embeddings from modernBERT with text features
    (URL, EMAIL, Phone) for detecting scammming messages
    """)
    st.divider()
    st.caption("Labels: Normal · Spam · Smishing")