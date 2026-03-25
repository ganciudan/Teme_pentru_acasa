import json
import os
import hashlib

from dotenv import load_dotenv, find_dotenv
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import faiss

load_dotenv(find_dotenv())

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "data_chunks.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
FAISS_META_PATH = os.path.join(DATA_DIR, "faiss.index.meta")
RESTAURANTE_JSON_PATH = os.path.join(DATA_DIR, "restaurante_iasi.json")
USE_MODEL_URL = os.environ.get(
    "USE_MODEL_URL",
    "https://tfhub.dev/google/universal-sentence-encoder/4",
)

WEB_URLS = [u for u in os.environ.get("WEB_URLS", "").split(";") if u]

class RAGAssistant:
    """Asistent RAG specializat pe restaurante din Iași, Romania."""

    def __init__(self) -> None:
        """Initializeaza clientul LLM, embedderul si prompturile."""
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Seteaza GROQ_API_KEY in variabilele de mediu.")

        self.client = OpenAI(
            api_key=self.groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        os.makedirs(DATA_DIR, exist_ok=True)
        self.embedder = None

        # Propozitie de referinta pentru domeniul restaurante Iași
        self.relevance = self._embed_texts(
            "Aceasta este o intrebare despre restaurante, localuri, meniuri, "
            "mancare, bauturi, preturi sau recomandari culinare in orasul Iasi, Romania.",
        )[0]

        # System prompt detaliat pentru ghidarea LLM-ului
        self.system_prompt = (
            "Esti un asistent specializat in gastronomia si restaurantele din Iasi, Romania. "
            "Raspunzi doar la intrebari despre restaurante, cafenele, baruri si localuri din Iasi: "
            "recomandari culinare, meniuri, preturi, locatii, atmosfera, evenimente, rezervari. "
            "Cand recomanzi restaurante, include intotdeauna pentru fiecare: "
            "numele complet, adresa exacta (strada, numar, zona din Iasi), "
            "tipul de bucatarie, intervalul de preturi (in lei sau €/€€/€€€), "
            "program (daca e disponibil), si informatii de contact sau website. "
            "Foloseste informatiile din contextul furnizat. Daca nu gasesti informatia in context, "
            "spune clar ca nu ai date suficiente despre acel restaurant sau subiect. "
            "Raspunde intotdeauna in limba romana, clar, concis si prietenos. "
            "Adapteaza-ti raspunsul la bugetul si preferintele utilizatorului (romantic, familie, prieteni). "
            "Daca utilizatorul intreaba despre facilitati (parcare, terasa, wifi, carduri), mentioneaza-le daca sunt in context."
        )


    def _load_documents_from_web(self) -> list[str]:
        """Incarca si chunked documente de pe site-uri prin WebBaseLoader."""
        if os.path.exists(CHUNKS_JSON_PATH):
            try:
                with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached, list) and cached:
                    return cached
            except (OSError, json.JSONDecodeError):
                pass

        all_chunks = []
        for url in WEB_URLS:
            try:
                print(f"[DEBUG] Loading {url} ...")
                loader = WebBaseLoader(url)
                docs = loader.load()
                print(f"[DEBUG] Got {len(docs)} doc(s) from {url}")
                for doc in docs:
                    print(f"[DEBUG] Doc content length: {len(doc.page_content)} chars")
                    chunks = self._chunk_text(doc.page_content)
                    print(f"[DEBUG] Chunks produced: {len(chunks)}")
                    all_chunks.extend(chunks)
            except Exception as e:
                print(f"[DEBUG] Failed to load {url}: {e}")
                continue

        print(f"[DEBUG] Total chunks from web: {len(all_chunks)}")

        # Incarca restaurante din fisierul JSON local
        local_chunks = self._load_from_local_json()
        all_chunks.extend(local_chunks)

        if all_chunks:
            with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False)

        return all_chunks

    def _load_from_local_json(self) -> list[str]:
        """Incarca restaurante din fisierul JSON local restaurante_iasi.json."""
        if not os.path.exists(RESTAURANTE_JSON_PATH):
            print(f"[DEBUG] Local JSON not found: {RESTAURANTE_JSON_PATH}")
            return []
        try:
            with open(RESTAURANTE_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[DEBUG] Failed to load local JSON: {e}")
            return []

        restaurante = data.get("restaurante", []) if isinstance(data, dict) else data
        chunks = []
        for r in restaurante:
            if not isinstance(r, dict):
                continue
            parts = []
            if r.get("nume"):
                parts.append(f"Restaurant: {r['nume']}")
            if r.get("adresa"):
                parts.append(f"Adresa: {r['adresa']}, {r.get('zona', '')}, Iasi")
            if r.get("tip_bucatarie"):
                parts.append(f"Tip bucatarie: {r['tip_bucatarie']}")
            if r.get("interval_preturi"):
                parts.append(f"Preturi: {r['interval_preturi']}")
            if r.get("pret_mediu_lei"):
                parts.append(f"Pret mediu: {r['pret_mediu_lei']} lei/persoana")
            if r.get("program"):
                parts.append(f"Program: {r['program']}")
            if r.get("telefon"):
                parts.append(f"Telefon: {r['telefon']}")
            if r.get("website"):
                parts.append(f"Website: {r['website']}")
            if r.get("facilitati"):
                facilitati_str = ", ".join(r['facilitati'])
                parts.append(f"Facilitati: {facilitati_str}")
            if r.get("vegetarian"):
                parts.append(f"Optiuni vegetariene: {'Da' if r['vegetarian'] else 'Nu'}")
            if r.get("sursa_url"):
                parts.append(f"Sursa: {r['sursa_url']}")
            if parts:
                chunks.append("\n".join(parts))
        print(f"[DEBUG] Local JSON chunks: {len(chunks)}")
        return chunks

    def _send_prompt_to_llm(
        self,
        user_input: str,
        context: str
    ) -> str:
        """Trimite promptul catre LLM si returneaza raspunsul."""

        system_msg = self.system_prompt

        # User prompt structurat pentru domeniul restaurante
        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": (
                    f"Folosind urmatorul context despre restaurantele din Iasi:\n\n"
                    f"{context}\n\n"
                    f"Raspunde la urmatoarea intrebare despre restaurante, meniuri, "
                    f"preturi, locatii sau recomandari culinare in Iasi.\n\n"
                    f"Pentru fiecare restaurant mentionat, include:\n"
                    f"- Nume restaurant\n"
                    f"- Adresa completa (strada, numar, zona)\n"
                    f"- Tip bucatarie (romaneasca, italiana, asiatica, etc.)\n"
                    f"- Interval preturi (€, €€, €€€ sau lei/persoana)\n"
                    f"- Program (daca e disponibil)\n"
                    f"- Contact/Website (daca e disponibil)\n"
                    f"- Facilitati relevante (parcare, terasa, rezervari, etc.)\n\n"
                    f"Intrebarea utilizatorului: {user_input}\n\n"
                    f"Daca contextul contine link-uri sau surse, include-le pentru referinta."
                ),
            },
        ]

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",  # Model mai bun decat gpt-oss-20b
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[DEBUG] LLM error: {e}")
            return (
                "Asistent: Nu pot ajunge la modelul de limbaj acum. "
                "Te rog incearca din nou in cateva momente."
            )
        
    def _embed_texts(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        """Genereaza embeddings folosind Universal Sentence Encoder."""
        if isinstance(texts, str):
            texts = [texts]
        if self.embedder is None:
            self.embedder = hub.load(USE_MODEL_URL)
        if callable(self.embedder):
            embeddings = self.embedder(texts)
        else:
            infer = self.embedder.signatures.get("default")
            if infer is None:
                raise ValueError("Model USE nu expune semnatura 'default'.")
            outputs = infer(tf.constant(texts))
            embeddings = outputs.get("default")
            if embeddings is None:
                raise ValueError("Model USE nu a returnat cheia 'default'.")
        return np.asarray(embeddings, dtype="float32")

    def _chunk_text(self, text: str) -> list[str]:
        """Imparte textul in bucati cu RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
        )
        chunks = splitter.split_text(text or "")
        return chunks if chunks else [""]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculeaza similaritatea cosine intre doi vectori."""
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _build_faiss_index_from_chunks(self, chunks: list[str]) -> faiss.IndexFlatIP:
        """Construieste index FAISS din chunks text si il salveaza pe disc."""
        if not chunks:
            raise ValueError("Lista de chunks este goala.")

        embeddings = self._embed_texts(chunks).astype("float32")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
            f.write(self._compute_chunks_hash(chunks))
        return index

    def _compute_chunks_hash(self, chunks: list[str]) -> str:
        """Hash determinist pentru lista de chunks si model."""
        payload = json.dumps(
            {
                "model": USE_MODEL_URL,
                "chunks": chunks,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_index_hash(self) -> str | None:
        """Incarca hash-ul asociat indexului FAISS."""
        if not os.path.exists(FAISS_META_PATH):
            return None
        try:
            with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return None

    def _retrieve_relevant_chunks(self, chunks: list[str], user_query: str, k: int = 10) -> list[str]:
        """Rankeaza chunks folosind FAISS si returneaza top-k relevante."""
        if not chunks:
            return []

        current_hash = self._compute_chunks_hash(chunks)
        stored_hash = self._load_index_hash()

        query_embedding = self._embed_texts(user_query).astype("float32")

        index = None
        if os.path.exists(FAISS_INDEX_PATH) and stored_hash == current_hash:
            try:
                index = faiss.read_index(FAISS_INDEX_PATH)
                if index.ntotal != len(chunks) or index.d != query_embedding.shape[1]:
                    index = None
            except Exception:
                index = None

        if index is None:
            index = self._build_faiss_index_from_chunks(chunks)

        faiss.normalize_L2(query_embedding)

        k = min(k, len(chunks))
        if k == 0:
            return []

        _, indices = index.search(query_embedding, k=k)
        return [chunks[i] for i in indices[0] if i < len(chunks)]

    def _retrieve_by_zone(self, user_query: str) -> list[str]:
        """Cauta direct in restaurante_iasi.json dupa zona sau adresa."""
        if not os.path.exists(RESTAURANTE_JSON_PATH):
            return []
        try:
            with open(RESTAURANTE_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return []

        restaurante = data.get("restaurante", []) if isinstance(data, dict) else data
        query_lower = user_query.lower()
        
        # Extrage keywords relevante (min 3 caractere)
        clean = "".join(c if c.isalpha() or c.isspace() else " " for c in query_lower)
        keywords = [w for w in clean.split() if len(w) >= 3]
        print(f"[DEBUG] Zone keywords: {keywords}")

        # Zone comune in Iasi
        zone_keywords = ["copou", "tatarasi", "palas", "centru", "pacurari", "tudor", 
                         "galata", "frumoasa", "nicolina", "tomesti", "ciric", "dacia"]
        
        matched = []
        for r in restaurante:
            zona = (r.get("zona") or "").lower()
            adresa = (r.get("adresa") or "").lower()
            
            # Match pe zona sau adresa
            if any(kw in zona or kw in adresa for kw in keywords if kw in zone_keywords or len(kw) >= 4):
                parts = []
                if r.get("nume"):
                    parts.append(f"Restaurant: {r['nume']}")
                if r.get("adresa"):
                    parts.append(f"Adresa: {r['adresa']}, {r.get('zona', '')}, Iasi")
                if r.get("tip_bucatarie"):
                    parts.append(f"Tip bucatarie: {r['tip_bucatarie']}")
                if r.get("interval_preturi"):
                    parts.append(f"Preturi: {r['interval_preturi']}")
                if r.get("pret_mediu_lei"):
                    parts.append(f"Pret mediu: {r['pret_mediu_lei']} lei/persoana")
                if r.get("program"):
                    parts.append(f"Program: {r['program']}")
                if r.get("telefon"):
                    parts.append(f"Telefon: {r['telefon']}")
                if r.get("website"):
                    parts.append(f"Website: {r['website']}")
                if r.get("facilitati"):
                    parts.append(f"Facilitati: {', '.join(r['facilitati'])}")
                if r.get("sursa_url"):
                    parts.append(f"Sursa: {r['sursa_url']}")
                if parts:
                    matched.append("\n".join(parts))
        
        print(f"[DEBUG] Zone match chunks: {len(matched)}")
        return matched

    def calculate_similarity(self, text: str) -> float:
        """Returneaza similaritatea cu propozitia de referinta despre restaurante Iasi."""
        embedding = self._embed_texts(text.strip())[0]
        return self._cosine_similarity(embedding, self.relevance)

    def is_relevant(self, user_input: str) -> bool:
        """Verifica daca intrarea utilizatorului e despre restaurante/gastronomie Iasi."""
        return self.calculate_similarity(user_input) >= 0.45  # Prag ajustat pentru flexibilitate

    def assistant_response(self, user_message: str) -> str:
        """Directioneaza mesajul utilizatorului catre calea potrivita."""
        if not user_message:
            return (
                "Te rog scrie o intrebare despre restaurantele din Iasi. "
                "Exemple: 'Care sunt cele mai bune restaurante italienesti din Iasi?', "
                "'Unde pot manca traditional in Copou?', "
                "'Recomanda-mi un restaurant romantic pentru cina cu buget mediu.'"
            )

        if not self.is_relevant(user_message):
            return (
                "Intrebarea ta nu pare a fi despre restaurante sau gastronomie din Iasi. "
                "Sunt specializat in recomandari culinare si informatii despre localurile din Iasi. "
                "Exemple de intrebari la care pot raspunde:\n"
                "- 'Care sunt cele mai bune restaurante din Tatarasi?'\n"
                "- 'Unde gasesc sushi bun in Iasi?'\n"
                "- 'Recomanda-mi o cafenea pentru studiat in centru.'\n"
                "- 'Care e pretul mediu la un pranz in Palas?'"
            )

        chunks = self._load_documents_from_web()
        relevant_chunks = self._retrieve_relevant_chunks(chunks, user_message, k=10)
        zone_chunks = self._retrieve_by_zone(user_message)
        
        # Merge si deduplica (zone matches first pentru prioritate)
        seen = set()
        combined = []
        for c in zone_chunks + relevant_chunks:
            if c not in seen:
                seen.add(c)
                combined.append(c)
        
        context = "\n\n".join(combined)
        return self._send_prompt_to_llm(user_message, context)

if __name__ == "__main__":
    assistant = RAGAssistant()
    
    # Test relevant - recomandari generale
    print("=" * 80)
    print("TEST 1: Recomandari restaurante romanesti Copou")
    print("=" * 80)
    print(assistant.assistant_response(
        "Care sunt cele mai bune restaurante romanesti din zona Copou?"
    ))
    
    print("\n" + "=" * 80)
    print("TEST 2: Restaurant specific - sushi")
    print("=" * 80)
    print(assistant.assistant_response(
        "Unde pot gasi sushi bun in Iasi cu buget mediu?"
    ))
    
    print("\n" + "=" * 80)
    print("TEST 3: Restaurant romantic cu buget")
    print("=" * 80)
    print(assistant.assistant_response(
        "Recomanda-mi un restaurant romantic pentru cina cu buget maxim 200 lei pentru 2 persoane"
    ))
    
    print("\n" + "=" * 80)
    print("TEST 4: Intrebare irelevanta")
    print("=" * 80)
    print(assistant.assistant_response(
        "Care este capitala Frantei?"
    ))
    
    print("\n" + "=" * 80)
    print("TEST 5: Cafenea pentru studiat")
    print("=" * 80)
    print(assistant.assistant_response(
        "Recomanda-mi o cafenea linistita pentru studiat in centru cu wifi bun"
    ))
