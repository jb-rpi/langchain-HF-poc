import os
import random
import textwrap
from dotenv import load_dotenv
from huggingface_hub import list_models, InferenceClient
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

# ================== Familles de modèles ==================
FAMILIES = {
    "qwen": "Qwen",
    "llama": "Llama",
    "deepseek": "deepseek",
    "mistral": "Mistral",
    "phi": "phi",
}


# ================== Sélection d'un modèle par famille ==================
def get_one_model_from_family(keyword: str):
    models_gen = list_models(
        token=HF_TOKEN,
        search=keyword,
        filter="text-generation",
        sort="downloads",
        direction=-1,
        limit=10,
    )
    first = next(models_gen, None)
    if first is None:
        return None
    return first.modelId


def pick_random_model():
    candidates = []

    print("Recherche de modèles par famille...\n")
    for family, keyword in FAMILIES.items():
        model = get_one_model_from_family(keyword)
        if model:
            candidates.append(model)
            print(f"[OK] Modèle trouvé pour {family}: {model}")
        else:
            print(f"[WARN] Aucun modèle trouvé pour la famille {family}")

    if len(candidates) < 2:
        raise RuntimeError(
            "Trop peu de modèles trouvés. Vérifie ta connexion ou ton token HF."
        )

    print("\n=== MODÈLES CANDIDATS ===")
    for m in candidates:
        print(" -", m)

    chosen = random.choice(candidates)
    print("\n=== MODÈLE TIRÉ AU SORT ===")
    print(chosen)

    return chosen


# ================== Fonction d'appel HF (chat) ==================
def build_hf_llm(model_id: str):
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

        return completion.choices[0].message.content

    return RunnableLambda(hf_chat_from_prompt)


# ================== Envoi Discord ==================
def send_to_discord(model_id: str, answer: str):
    # On tronque si la réponse est trop longue (limite Discord ~2000 chars par message)
    max_len = 1900
    short_answer = answer if len(answer) <= max_len else answer[:max_len] + "\n[...]"

    content = textwrap.dedent(f"""
    **Rapport triglycérides / cholestérol — LLM Hugging Face**

    Modèle utilisé : `{model_id}`

    Réponse :
    ```markdown
    {short_answer}
    ```
    """)

    resp = requests.post(
        DISCORD_WEBHOOK_URL,
        json={"content": content},
        timeout=10,
    )
    resp.raise_for_status()


# ================== Programme principal ==================
def main():
    chosen_model = pick_random_model()
    hf_llm = build_hf_llm(chosen_model)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Tu fournis uniquement des recommandations générales basées sur des "
                    "informations publiques. Tu n’es pas un médecin. Tu rappelles "
                    "toujours que cela ne remplace pas un avis médical et qu'il faut "
                    "consulter un professionnel de santé pour un avis personnalisé."
                ),
            ),
            (
                "human",
                (
                    "Donne-moi des recommandations concrètes pour faire baisser à la fois "
                    "les triglycérides et le cholestérol (LDL), avec un focus sur : "
                    "alimentation, activité physique, hygiène de vie, et habitudes à éviter. "
                    "Propose une to-do list en 7 points maximum, claire et actionnable. "
                    "MAXIMUM 200 MOTS."
                ),
            ),
        ]
    )

    chain = prompt | hf_llm | StrOutputParser()

    print("\n=== QUESTION ENVOYÉE AU MODÈLE ===")
    print(f"Modèle utilisé : {chosen_model}\n")

    answer = chain.invoke({})

    print("=== RÉPONSE DU MODÈLE ===\n")
    print(answer)

    print("\nEnvoi de la réponse sur Discord...")
    send_to_discord(chosen_model, answer)
    print("OK.")


if __name__ == "__main__":
    main()
