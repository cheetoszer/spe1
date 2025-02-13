import gradio as gr
import openai
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from numpy.linalg import norm
from typing import Tuple, List, Dict, Generator


openai.api_key = "sk-no-key-required"
openai.api_base = "http://localhost:8080/v1"




def load_embedding_model(model_name: str = "intfloat/multilingual-e5-large-instruct") -> Tuple[AutoTokenizer, AutoModel, str]:
    """
    Charge le modèle d'embedding et son tokenizer.
    
    Paramètres :
    model_name (str) : Nom du modèle à charger.
    
    Retourne :
    Tuple contenant le tokenizer, le modèle et le périphérique utilisé (CPU/GPU).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_embedding_model()




#liste des différentes catégorie de la faq
candidates = [
    "performances_generales",
    "competences_cognitives",
    "types_operations",
    "amelioration",
    "analyse_comparative",
    "stats_detaillees",
    "strategie",
    "fallback"
]




def vectorize_query(query: str, tokenizer: AutoTokenizer, model: AutoModel, device: str) -> np.ndarray:
    """
    Convertit une requête en un vecteur d'embedding.
    
    Paramètres :
    query (str) : La requête utilisateur.
    tokenizer (AutoTokenizer) : Tokenizer du modèle de transformation.
    model (AutoModel) : Modèle de transformation.
    device (str) : Périphérique utilisé (CPU/GPU).
    
    Retourne :
    np.ndarray : Vecteur d'embedding de la requête.
    """
    query = f"query: {query}"
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy()




#calculer les embeddings une fois pour éviter le faire à chaque appel de classify_query()
candidates_emb = [vectorize_query(c, tokenizer, model, device) for c in candidates] 




def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calcule la similarité cosinus entre deux vecteurs numpy.
    
    Paramètres :
    vec_a (np.ndarray) : Premier vecteur.
    vec_b (np.ndarray) : Deuxième vecteur.
    
    Retourne :
    float : Score de similarité cosinus entre les deux vecteurs.
    """
    a = vec_a.flatten()
    b = vec_b.flatten()
    dot_prod = np.dot(a, b)
    denom = norm(a) * norm(b)
    return 0.0 if denom == 0 else dot_prod / denom




def classify_query(user_input: str, candidates: List[str], tokenizer: AutoTokenizer, model: AutoModel, device: str) -> str:
    """
    Classifie une requête utilisateur en identifiant la catégorie la plus pertinente.
    
    Paramètres :
    user_input (str) : Requête de l'utilisateur.
    candidates (List[str]) : Liste des catégories possibles.
    tokenizer (AutoTokenizer) : Tokenizer du modèle de transformation.
    model (AutoModel) : Modèle de transformation.
    device (str) : Périphérique utilisé (CPU/GPU).
    
    Retourne :
    str : Catégorie la plus proche de la requête utilisateur.
    """
    user_embedding = vectorize_query(user_input, tokenizer, model, device)
    similarities = [compute_cosine_similarity(user_embedding, candidate_emb) for candidate_emb in candidates_emb]
    best_idx = np.argmax(similarities)
    return candidates[best_idx]




def get_role_prompt(query: str) -> str:
    """
    Génère un prompt de rôle personnalisé en fonction de la catégorie de la requête utilisateur.
    
    Paramètres :
    query (str) : Requête de l'utilisateur.
    
    Retourne :
    str : Prompt de rôle associé.
    """
    candidate = classify_query(query, candidates, tokenizer, model, device)
    roles = {
        "performances_generales": "Tu es l'assistant d'un jeu de mathématiques. \nTa spécialité est de répondre aux questions sur les PERFORMANCES GLOBALES de l’utilisateur : \n- taux de réussite,\n- évolution des scores,\n- niveaux atteints,\n- etc.",
        "competences_cognitives": "Tu es l'assistant d’un jeu de mathématiques, \nspécialisé dans l'évaluation des COMPÉTENCES COGNITIVES (mémoire de travail, vitesse de traitement, etc.).",
        "types_operations": "Tu es l'assistant d'un jeu de mathématiques, \nfocalisé sur les TYPES D’OPÉRATIONS (multiplication, division, addition, puissance, etc.). \nTu aides l’utilisateur à identifier ses points forts et ses faiblesses dans chaque type d’opération.",
        "amelioration": "Tu es l'assistant d’un jeu de mathématiques. \nTa priorité est de fournir des CONSEILS D’AMÉLIORATION, \nen abordant les techniques de calcul mental, les objectifs de temps, et les entraînements ciblés.",
        "analyse_comparative": "Tu es l'assistant d’un jeu de mathématiques. \nIci, tu compares les performances PASSÉES vs. ACTUELLES de l’utilisateur \n(en termes de taux de réussite, vitesse de réponse, score, etc.).",
        "stats_detaillees": "Tu es l’assistant d’un jeu de mathématiques, \napte à fournir des STATISTIQUES DÉTAILLÉES sur les dernières sessions de l’utilisateur \n(score moyen, temps moyen, opérations les plus réussies, etc.).",
        "strategie": "Tu es l'assistant d’un jeu de mathématiques. \nTon rôle est de proposer des STRATÉGIES concrètes pour optimiser le score, \nla rapidité de calcul et la progression globale de l’utilisateur.",
        "fallback": "Tu es l’assistant d’un jeu de mathématiques."
    }
    return roles.get(candidate, roles["fallback"])




def get_prompt_system(query: str) -> str:
    """
    Génère un prompt système structuré en fonction de la requête utilisateur.
    
    Paramètres :
    query (str) : Requête de l'utilisateur.
    
    Retourne :
    str : Prompt formaté pour guider la réponse du modèle.
    """

    role = get_role_prompt(query)
    system_prompt = f"""

    {role}
    
    - Tu adoptes un ton pédagogue, sérieux, avec une pointe d’humour pour rester engageant.
    - Tu fais preuve de bienveillance, d’encouragement et d’empathie : l’utilisateur doit se sentir valorisé et motivé.
    - Tu t’adaptes au registre de langage de l’utilisateur :
        - S’il tutoie, tu tutoies.
        - S’il vouvoie, tu vouvoies.
        - S’il emploie un langage familier ou grossier, tu peux être plus direct tout en restant courtois.

    RÈGLES DE CONTENU
    - Logique du jeu : tu peux expliquer librement les règles et le fonctionnement interne du jeu si l’utilisateur le demande.
    - Hors-sujet : si la question n’est pas liée au jeu, réponds de manière concise que tu n’as pas accès à ces informations.
    - Il n’existe pas de support externe vers lequel renvoyer l’utilisateur.
    - Pas d’informations en base : si la question est pertinente mais que tu ne disposes pas de données précises, propose une réponse générale ou des conseils issus de tes connaissances.

    FORMAT
    - Réponds de façon concise, avec un style clair et structuré.
    - Réponds en Français.
    - Utilise des listes à puce lorsque c’est pertinent pour organiser l’information (chiffres, conseils, étapes, plan…).

    OBJECTIF
    - Aider l’utilisateur à comprendre et améliorer ses performances mathématiques et cognitives.
    - Mettre en avant ses points forts et proposer des conseils concrets pour qu’il progresse, tout en le motivant.
    """
    return system_prompt




def search_qdrant(query_vector: np.ndarray, client: QdrantClient, collection_name: str, top_k: int = 3) -> List[Dict]:
    """
    Effectue une recherche de similarité dans Qdrant.
    
    Paramètres :
    query_vector (np.ndarray) : Vecteur de la requête utilisateur.
    client (QdrantClient) : Client Qdrant.
    collection_name (str) : Nom de la collection à interroger.
    top_k (int) : Nombre de résultats à récupérer.
    
    Retourne :
    List[Dict] : Liste des documents les plus pertinents.
    """
    return client.search(collection_name=collection_name, query_vector=query_vector[0].tolist(), limit=top_k, with_payload=True)




def retrieve_and_generate_answer(query: str, collection_name: str = "GameRag") -> Generator[str, None, None]:
    """
    Récupère le contexte pertinent via Qdrant et génère une réponse en streaming.
    
    Paramètres :
    query (str) : Requête utilisateur.
    collection_name (str) : Nom de la collection dans Qdrant.
    
    Retourne :
    Generator[str, None, None] : Réponse générée progressivement.
    """

    qdrant_client = QdrantClient(url="http://localhost:6333")

    query_vector = vectorize_query(query, tokenizer, model, device)
    results = search_qdrant(query_vector, qdrant_client, collection_name)
    retrieved_texts = [r.payload.get("texte", "") for r in results]
    context = "\n\n".join(retrieved_texts)

    prompt_system = get_prompt_system(query)

    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
    ]

    response = openai.ChatCompletion.create(
        model="mistral-7b-instruct",
        messages=messages,
        temperature=0.7,
        top_p=0.9,
        stream=True
    )
    print(prompt_system)
    full_response = ""
    for chunk in response:
        if "content" in chunk["choices"][0]["delta"]:
            token = chunk["choices"][0]["delta"]["content"]
            full_response += token
            yield full_response





iface = gr.Interface(
    fn=retrieve_and_generate_answer,
    inputs=gr.Textbox(label="Posez votre question"),
    outputs=gr.Textbox(label="Réponse générée", interactive=True),
    title="Game Rag",
    description="Quelle est votre question ?",
)




iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
