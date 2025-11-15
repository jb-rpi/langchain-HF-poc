# main.py
import os
from dotenv import load_dotenv

from huggingface_hub import InferenceClient

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# ========= Charger .env =========
# .env doit contenir au minimum :
# HF_TOKEN=xxxxxxxx
# HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct  (optionnel, sinon valeur par défaut ci-dessous)
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

if not HF_TOKEN:
    raise RuntimeError(
        "Le token Hugging Face (HF_TOKEN) est manquant dans le fichier .env"
    )

# ========= Client Hugging Face (chat) =========
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)


def hf_chat_from_prompt(prompt_value):
    """
    prompt_value est un ChatPromptValue produit par ChatPromptTemplate.
    On le convertit en liste de messages pour l'API chat HF,
    puis on renvoie le texte de la réponse.
    """
    messages = prompt_value.to_messages()

    hf_messages = []
    for m in messages:
        if m.type == "system":
            role = "system"
        elif m.type == "human":
            role = "user"
        elif m.type == "ai":
            role = "assistant"
        else:
            role = "user"

        # m.content peut être soit une string, soit une liste de blocks;
        # ici on gère le cas le plus simple (string).
        hf_messages.append(
            {
                "role": role,
                "content": str(m.content),
            }
        )

    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=hf_messages,
        max_tokens=400,
        temperature=0.7,
        top_p=0.9,
    )

    return completion.choices[0].message.content


# On enveloppe la fonction ci-dessus dans un Runnable pour LangChain
hf_llm = RunnableLambda(hf_chat_from_prompt)

# ========= Prompt =========
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Tu fournis uniquement des recommandations générales basées sur des "
                "informations publiques. Tu n’es pas un médecin. Tu rappelles "
                "systématiquement de consulter un professionnel de santé pour un avis "
                "personnalisé."
            ),
        ),
        (
            "human",
            (
                "Donne-moi des recommandations actionnables pour faire baisser les "
                "triglycérides (alimentation, activité physique, hygiène de vie). "
                "Limite-toi à 7 points structurés et ajoute un rappel final de consulter "
                "un médecin."
            ),
        ),
    ]
)

# ========= Chaîne LangChain =========
chain = prompt | hf_llm | StrOutputParser()

if __name__ == "__main__":
    print("=== Requête envoyée ===\n")

    answer = chain.invoke({})
    print("=== Réponse du modèle ===\n")
    print(answer)
