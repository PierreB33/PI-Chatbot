# %%
import requests  
import json  
from sentence_transformers import SentenceTransformer 
import faiss  
from fuzzywuzzy import fuzz, process  
import numpy as np

# %% [markdown]
# # Modèle gemini

# %%
API_KEY = 'AIzaSyDiOkmD77M8RIG2bGJh034IlSEe9iIq0H4'
API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent'
headers = {'Content-Type': 'application/json'}
params = {'key': API_KEY}

# %%
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# %%
import os
import requests
from bs4 import BeautifulSoup
import easyocr

reader = easyocr.Reader(['en', 'fr']) 

def scrape_website_with_images(url, image_folder="images"):

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Erreur lors de l'accès au site : {response.status_code}")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    text_content = ' '.join([p.get_text() for p in soup.find_all('p')])
    
    images = soup.find_all('img')
    image_texts = []
    
    for img in images:
        img_url = img.get('src')
        if not img_url:
            continue
        
        try:
            if not img_url.startswith("http"):  
                img_url = url + img_url
            
            img_response = requests.get(img_url, stream=True)
            if img_response.status_code == 200:
                img_name = os.path.join(image_folder, os.path.basename(img_url))
                with open(img_name, 'wb') as f:
                    f.write(img_response.content)
                
                img_text = reader.readtext(img_name, detail=0)
                image_texts.append(" ".join(img_text))
        except Exception as e:
            print(f"Erreur lors du traitement de l'image {img_url}: {e}")
    
    full_content = text_content + " " + " ".join(image_texts)
    return full_content



# %%
def embed_content(content, model):
 
    embedding = model.encode(content)
    return embedding

# %%
def create_faiss_index(embeddings):
   
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# %%
def load_faiss_index(filepath, dimension):
    index = faiss.IndexFlatL2(dimension)
    return faiss.read_index(filepath)

# %%
def save_embeddings(embeddings, filepath):

    with open(filepath, 'w') as f:
        json.dump(embeddings.tolist(), f) 

def save_faiss_index(index, filepath):

    faiss.write_index(index, filepath)


# %%
def save_documents(documents, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)


# %%
documents = []

def process_website(url, model, embeddings_path, index_path,documents_path):

    content = scrape_website_with_images(url)
    documents.append(content)
    
    embedding = embed_content(content, model)
    embeddings = np.array([embedding])
    
    index = create_faiss_index(embeddings)

    save_documents(documents, documents_path)
    save_embeddings(embeddings, embeddings_path)
    save_faiss_index(index, index_path)
    
    print("Traitement terminé !")


# %%
#Appel du pipeline permettant scrapping, embedding, indexage et sauvegarde des docs

documents = []  
documents_path = "documents.json"
embeddings_path = "embeddings.json"
index_path = "faiss_index.bin"

urls = [
    "https://faq.manaos.com/manaos-faq/what-is-manaos",
    "https://faq.manaos.com/manaos-faq/what-are-our-values",
    "https://faq.manaos.com/manaos-faq/how-does-look-through-work"
]

for url in urls:
    process_website(url, embedding_model, embeddings_path, index_path, documents_path)

print("Tous les sites ont été traités et sauvegardés.")



# %%
def retrieve(query, model, index, top_k=1, similarity_threshold=None):

    query_embedding = model.encode(query).reshape(1, -1)
    
    distances, indices = index.search(query_embedding, top_k)
    
    retrieved_docs = []
    for idx, dist in zip(indices[0], distances[0]):
        if similarity_threshold is None or dist <= similarity_threshold:
            retrieved_docs.append({
                "document": idx,  
                "distance": dist  
            })
    
    return retrieved_docs


# %%
def query_gemini(query, context=""):
    prompt = f"Contexte : {context}\nQuestion : {query}\nRéponse :"
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    response = requests.post(API_URL, headers=headers, params=params, data=json.dumps(data))
    
    if response.status_code == 200:
        result = response.json()
        print("Réponse complète de Gemini :", result)  
        try:
            response_text = result['candidates'][0]['content']['parts'][0]['text']
            return response_text
        except (IndexError, KeyError):
            return 'Pas de réponse générée par Gemini'
    else:
        print(f"Erreur Gemini: {response.status_code}, {response.text}")
        return None

# %%
def generate_response_with_rag_gemini(query,index, similarity_threshold=50, top_k=1):
    retrieved_docs = retrieve(query, model=embedding_model, index=index, top_k=top_k, similarity_threshold=similarity_threshold)

    if retrieved_docs:
        print("Utilisation du RAG")
        context = " ".join([documents[doc['document']] for doc in retrieved_docs])
        print("Contexte utilisé pour RAG :", context)
    else:
        print("Aucun document pertinent trouvé.")
        context = ""
        
    response = query_gemini(query, context)
    return response


# %%
index = load_faiss_index("faiss_index.bin", 768)

query = "What are the 2components on which Manaos is based ? "
response = generate_response_with_rag_gemini(query,index)
print("Question:", query)
print("Réponse générée:", response)


