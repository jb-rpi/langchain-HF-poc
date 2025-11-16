import os
import random
import textwrap
import re
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import requests

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# ================== Chargement .env ==================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN manquant dans .env")

if not DISCORD_WEBHOOK_URL:
    raise RuntimeError("DISCORD_WEBHOOK_URL manquant dans .env")


# ================== Liste blanche de modèles ==================
# Si un modèle pose problème, commente-le dans cette liste.
MODELS = [
    "deepseek-ai/DeepSeek-OCR",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-0528",
    "deepseek-ai/DeepSeek-V3.1",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "deepseek-ai/DeepSeek-V3.2-Exp",
    "deepseek-ai/DeepSeek-V3.1-Terminus",
]



# ================== Utils ==================
def strip_think_block(text: str) -> str:
    """
    Supprime un éventuel bloc <think>...</think> pour ne garder que la réponse finale.
    (utile pour certains modèles DeepSeek R1 qui ajoutent cette "pensée" interne).
    """
    clean = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return clean.strip() or text.strip()


def build_hf_llm(model_id: str):
    """
    Construit un Runnable LangChain qui appelle l'API Hugging Face
    via InferenceClient.chat.completions.
    """
    client = InferenceClient(model=model_id, token=HF_TOKEN)

    def hf_chat_from_prompt(prompt_value):
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

            hf_messages.append(
                {
                    "role": role,
                    "content": str(m.content),
                }
            )

        completion = client.chat.completions.create(
            model=model_id,
            messages=hf_messages,
            max_tokens=400,
            temperature=0.7,
            top_p=0.9,
        )

        raw = completion.choices[0].message.content or ""
        clean = strip_think_block(raw)
        return clean

    return RunnableLambda(hf_chat_from_prompt)


def send_to_discord(model_id: str, answer: str):
    """
    Envoie la réponse sur Discord via webhook.
    """
    max_len = 1800  # marge sous les 2000 caractères
    answer = answer.strip()
    short_answer = answer if len(answer) <= max_len else answer[:max_len] + "\n[...] (tronqué)"

    content = textwrap.dedent(f"""\
    **Rapport triglycérides / cholestérol — LLM DeepSeek (Hugging Face)**

    Modèle utilisé : `{model_id}`

    Réponse :
    {short_answer}
    """)

    resp = requests.post(
        DISCORD_WEBHOOK_URL,
        json={"content": content},
        timeout=10,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("Erreur Discord:", e)
        print("Réponse Discord:", resp.text)
        raise


# ================== Programme principal ==================
def main():
    if not MODELS:
        raise RuntimeError("La liste MODELS est vide.")

    # On mélange les modèles pour varier
    candidates = MODELS[:]
    random.shuffle(candidates)

    # Prompt commun
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Tu fournis uniquement des recommandations générales basées sur des "
                    "informations publiques. Tu es expert en nutrition et spécialisé en cardiologie, "
                    "mais ta réponse ne remplace en aucun cas un avis médical."
                ),
            ),
            (
                "human",
                (
                    "Donne-moi des recommandations concrètes pour faire baisser à la fois "
                    "les triglycérides et le cholestérol (LDL), avec un focus sur : "
                    "alimentation, activité physique, hygiène de vie, et habitudes à éviter. "
                    "Propose une to-do list en 7 points maximum, claire et actionnable. "
                    "MAXIMUM 1500 CARACTERES."
                ),
            ),
        ]
    )

    last_error = None

    for model_id in candidates:
        print("\n=== ESSAI AVEC LE MODÈLE ===")
        print(model_id)
        try:
            hf_llm = build_hf_llm(model_id)
            chain = prompt | hf_llm | StrOutputParser()

            print("\n=== QUESTION ENVOYÉE AU MODÈLE ===")
            print(f"Modèle utilisé : {model_id}\n")

            answer = chain.invoke({})
            print("=== RÉPONSE DU MODÈLE ===\n")
            print(answer)

            print("\nEnvoi de la réponse sur Discord...")
            send_to_discord(model_id, answer)
            print("OK.")
            return  # on sort après le premier modèle qui fonctionne
        except Exception as e:
            print(f"[WARN] Le modèle {model_id} a échoué: {e}")
            last_error = e

    # Si aucun modèle n'a marché
    raise RuntimeError(f"Aucun modèle DeepSeek n'a fonctionné. Dernière erreur: {last_error}")


if __name__ == "__main__":
    main()