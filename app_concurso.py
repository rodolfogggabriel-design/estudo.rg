import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import requests
import time
import re
import uuid
from collections import Counter

from multi_api_concurso import MultiAPIManager

load_dotenv()

app = Flask(__name__)

KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'kb_concurso.json')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
TOP_K = 3
MAX_HISTORY = 10

knowledge_base = None
api_manager = MultiAPIManager()

# =============================================
# PROMPTS — FOCO EM CONCURSO PÚBLICO
# =============================================
SYSTEM_PROMPT = """Voce e o estudo.RG, um assistente especializado em preparacao para concursos publicos brasileiros.
Voce responde perguntas com base no material de estudo fornecido como contexto.

REGRAS:
1. Responda SEMPRE em portugues brasileiro claro e objetivo
2. Base suas respostas EXCLUSIVAMENTE no contexto fornecido
3. Cite a fonte e a pagina quando referenciar informacoes especificas
4. Conecte o conteudo com o que costuma ser cobrado em provas (CESPE, FCC, FGV, VUNESP, etc.)
5. Use linguagem acessivel mas tecnicamente precisa
6. Quando aplicavel, cite artigos de lei, sumulas ou jurisprudencia presente no material
7. Se o contexto nao contiver informacao suficiente, informe claramente
8. Ao final, sugira topicos relacionados que costumam cair juntos em provas

CONTEXTO DO MATERIAL:
"""

FLASHCARD_PROMPT = """Com base EXCLUSIVAMENTE no contexto do material de concurso fornecido, gere exatamente 5 flashcards sobre o tema solicitado.
Os flashcards devem focar em definicoes, artigos de lei, principios e conceitos cobrados em provas.
FORMATO OBRIGATORIO (responda APENAS com este JSON, sem texto antes ou depois):
[{"frente":"pergunta ou enunciado de prova","verso":"resposta completa e objetiva","livro":"nome da fonte","pagina":"num"}]
Responda em portugues brasileiro. CONTEXTO:
"""

QUIZ_PROMPT = """Com base EXCLUSIVAMENTE no contexto do material fornecido, gere exatamente 5 questoes de multipla escolha no estilo de concurso publico sobre o tema solicitado.
Use estilo de bancas como CESPE (certo/errado adaptado para 4 alternativas) ou FCC/FGV (questoes dissertativas).
FORMATO OBRIGATORIO (responda APENAS com este JSON, sem texto antes ou depois):
[{"pergunta":"enunciado no estilo concurso","alternativas":["a) ...","b) ...","c) ...","d) ..."],"correta":0,"explicacao":"gabarito comentado citando a fonte","livro":"nome","pagina":"num"}]
Responda em portugues brasileiro. CONTEXTO:
"""

MINDMAP_PROMPT = """Com base EXCLUSIVAMENTE no contexto do material de concurso, gere um mapa mental sobre o tema solicitado.
Organize os conceitos como aparecem em provas: principios, definicoes, artigos importantes, excecoes.
FORMATO OBRIGATORIO (responda APENAS com este JSON):
{"centro":"tema","resumo":"resumo de 2-3 frases do tema com foco em concurso","ramos":[{"titulo":"subtema","cor":"#f59e0b","itens":["conceito A","conceito B"]}],"livros":["fonte, p.X"]}
Use 3-6 ramos, 2-5 itens cada. Cores: #f59e0b, #a78bfa, #60a5fa, #34d399, #ff6b9d, #fb923c. Portugues brasileiro. CONTEXTO:
"""

EXAM_MC_PROMPT = """Gere {n} questoes de MULTIPLA ESCOLHA nivel {nivel} sobre "{tema}" no estilo de concurso publico.
Use linguagem e formato identicos ao das bancas brasileiras (CESPE, FCC, FGV, VUNESP).
FORMATO JSON (sem texto extra):
[{{"pergunta":"Enunciado da questao no estilo banca","alternativas":["a)...","b)...","c)...","d)..."],"correta":0,"explicacao":"gabarito comentado com base legal ou doutrinaria","peso":{peso},"livro":"nome","pagina":"num","saiba_mais":"topico para aprofundar","sugestao_chat":"pergunta sugerida"}}]
Nivel facil=definicoes e conceitos basicos, medio=aplicacao e interpretacao, dificil=casos concretos e excecoes. Portugues brasileiro.
CONTEXTO:
"""

EXAM_DISC_PROMPT = """Gere {n} questoes DISCURSIVAS nivel {nivel} sobre "{tema}" no estilo de concurso publico.
Inclua questoes abertas como as cobradas em provas discursivas de cargos publicos.
FORMATO JSON (sem texto extra):
[{{"pergunta":"Enunciado da questao discursiva","resposta_esperada":"resposta modelo completa com base legal","pontos_chave":["ponto 1","ponto 2","ponto 3"],"peso":{peso},"livro":"nome","pagina":"num","saiba_mais":"topico para aprofundar","sugestao_chat":"pergunta sugerida"}}]
Portugues brasileiro.
CONTEXTO:
"""

EXAM_MATCH_PROMPT = """Gere {n} questoes de LIGAR COLUNAS nivel {nivel} sobre "{tema}" para concurso publico.
Coluna A: termos, artigos, institutos. Coluna B: definicoes, consequencias, fundamentos.
FORMATO JSON (sem texto extra):
[{{"instrucao":"Ligue os itens da coluna A com a coluna B","coluna_a":["item 1","item 2","item 3","item 4"],"coluna_b":["par 1","par 2","par 3","par 4"],"pares_corretos":[0,1,2,3],"peso":{peso},"explicacao":"explicacao com base legal","livro":"nome","pagina":"num","saiba_mais":"topico","sugestao_chat":"pergunta"}}]
pares_corretos: indice da coluna_b que corresponde a cada item da coluna_a.
Portugues brasileiro.
CONTEXTO:
"""

EXAM_ORDER_PROMPT = """Gere {n} questoes de ORDENAR SEQUENCIA nivel {nivel} sobre "{tema}" para concurso publico.
Use processos administrativos, fases procedimentais, hierarquias normativas.
FORMATO JSON (sem texto extra):
[{{"instrucao":"Ordene os itens/etapas na sequencia correta","itens_embaralhados":["item C","item A","item D","item B"],"ordem_correta":[1,3,0,2],"peso":{peso},"explicacao":"explicacao da ordem correta com base legal","livro":"nome","pagina":"num","saiba_mais":"topico","sugestao_chat":"pergunta"}}]
ordem_correta: indice que cada item embaralhado deve ocupar (0=primeiro, 1=segundo...).
Portugues brasileiro.
CONTEXTO:
"""

STUDY_GUIDE_PROMPT = """Com base EXCLUSIVAMENTE no contexto do material, gere um GUIA DE ESTUDO para concurso publico sobre o tema.
Organize como um resumo de caderno de concurseiro: pontos mais cobrados, pegadinhas, artigos importantes.
FORMATO JSON (sem texto extra):
{{"titulo":"Guia Concurso: tema","resumo":"resumo executivo de 3-4 frases com foco no que cai mais em prova","topicos":[{{"titulo":"topico 1","conteudo":"explicacao detalhada com base legal","importancia":"alta/media/baixa"}}],"termos_chave":[{{"termo":"palavra","definicao":"definicao juridica/tecnica"}}],"dicas_estudo":["dica 1 para concurso","dica 2"],"perguntas_frequentes":["questao que costuma cair 1","questao 2","questao 3"],"livros_referencia":["fonte, p.X"]}}
Portugues brasileiro.
CONTEXTO:
"""


# =============================================
# KNOWLEDGE BASE
# =============================================
def load_knowledge_base():
    global knowledge_base
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print("Base de conhecimento nao encontrada: " + KNOWLEDGE_BASE_PATH)
        knowledge_base = {"chunks": [], "books": [], "total_chunks": 0}
        return
    print("Carregando base de conhecimento...")
    with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
    for chunk in knowledge_base["chunks"]:
        if "embedding" in chunk and chunk["embedding"]:
            chunk["embedding"] = np.array(chunk["embedding"])
    print(str(knowledge_base['total_chunks']) + " chunks carregados")
    print("Materiais: " + ', '.join(knowledge_base.get('books', [])))


def cosine_similarity(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0
    return dot / (na * nb)


def search_similar_chunks(query_embedding, top_k=TOP_K, chunks=None):
    target = chunks if chunks else (knowledge_base.get("chunks", []) if knowledge_base else [])
    if not target:
        return []
    sims = []
    for chunk in target:
        emb = chunk.get("embedding")
        if emb is None:
            continue
        if not isinstance(emb, np.ndarray):
            emb = np.array(emb)
        sim = cosine_similarity(query_embedding, emb)
        sims.append((sim, chunk))
    sims.sort(key=lambda x: x[0], reverse=True)
    return [{"text": c["text"], "book": c.get("book", "Material"), "page": c.get("page", ""), "similarity": float(s)} for s, c in sims[:top_k]]


def keyword_search(query, top_k=TOP_K, chunks=None):
    target = chunks if chunks else (knowledge_base.get("chunks", []) if knowledge_base else [])
    if not target:
        return []
    stopwords = {'o','a','os','as','um','uma','de','do','da','dos','das','em','no','na','nos','nas','por','para','com','sem','que','qual','como','onde','quando','se','e','ou','mas','pois','porque','entre','sobre','mais','menos','muito','pouco','todo','toda','ser','ter','estar','fazer','pode','tem','sao','foi','eh','isso','este','esta','esse','essa','ao','pela','pelo'}
    qw = [w for w in re.findall(r'\w+', query.lower()) if w not in stopwords and len(w) > 2]
    if not qw:
        qw = re.findall(r'\w+', query.lower())
    scored = []
    for chunk in target:
        tl = chunk["text"].lower()
        score = sum(tl.count(w) for w in qw)
        if query.lower() in tl:
            score += 10
        for i in range(len(qw) - 1):
            if qw[i] + " " + qw[i+1] in tl:
                score += 5
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    mx = scored[0][0] if scored else 1
    return [{"text": c["text"], "book": c.get("book", "Material"), "page": c.get("page", ""), "similarity": round(min(s / max(mx, 1), 1.0) * 100) / 100} for s, c in scored[:top_k]]


def get_query_embedding(query):
    if GEMINI_API_KEY and GEMINI_API_KEY not in ("", "cole_sua_chave_aqui"):
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=" + GEMINI_API_KEY
            resp = requests.post(url, json={"model": "models/text-embedding-004", "content": {"parts": [{"text": query}]}}, timeout=10)
            if resp.status_code == 200:
                return np.array(resp.json()["embedding"]["values"])
        except:
            pass
    return None


def get_context_for_topic(topic, top_k=4, chunks=None):
    query_emb = get_query_embedding(topic)
    if query_emb is not None:
        results = search_similar_chunks(query_emb, top_k=top_k, chunks=chunks)
    else:
        results = keyword_search(topic, top_k=top_k, chunks=chunks)
    if not results:
        return "", []
    parts = []
    for i, c in enumerate(results, 1):
        parts.append("[Trecho " + str(i) + " - " + c['book'] + ", p." + str(c['page']) + "]\n" + c['text'])
    context = "\n\n---\n\n".join(parts)[:3000]
    seen = set()
    sources = []
    for r in results:
        key = r['book'] + "-p" + str(r['page'])
        if key not in seen:
            sources.append({"book": r["book"], "page": r["page"]})
            seen.add(key)
    return context, sources


def parse_json_response(text):
    text = text.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except:
        match = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None


# =============================================
# SUPABASE HELPERS
# =============================================
def supabase_request(method, path, data=None, token=None):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    headers = {
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    url = SUPABASE_URL + "/rest/v1/" + path
    try:
        if method == "GET":
            r = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            r = requests.post(url, headers=headers, json=data, timeout=10)
        elif method == "DELETE":
            r = requests.delete(url, headers=headers, timeout=10)
        elif method == "PATCH":
            r = requests.patch(url, headers=headers, json=data, timeout=10)
        else:
            return None
        if r.status_code < 300:
            return r.json() if r.text else {}
    except:
        pass
    return None


def get_user_from_token(req):
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None, None
    token = auth[7:]
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None, token
    try:
        r = requests.get(SUPABASE_URL + "/auth/v1/user", headers={"apikey": SUPABASE_KEY, "Authorization": "Bearer " + token}, timeout=10)
        if r.status_code == 200:
            return r.json(), token
    except:
        pass
    return None, token


# =============================================
# ROUTES
# =============================================
@app.route("/")
def index():
    books = knowledge_base.get("books", []) if knowledge_base else []
    total = knowledge_base.get("total_chunks", 0) if knowledge_base else 0
    return render_template("index.html", books=books, total_chunks=total,
                           supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY)


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()
    history = data.get("history", [])
    material_id = data.get("material_id")

    if not question:
        return jsonify({"error": "Pergunta vazia"}), 400

    start = time.time()
    chunks = None
    if material_id:
        user, token = get_user_from_token(request)
        if user and token:
            mat = supabase_request("GET", "materials?id=eq." + material_id + "&user_id=eq." + user["id"], token=token)
            if mat and len(mat) > 0:
                chunks = mat[0].get("chunks", [])

    query_emb = get_query_embedding(question)
    if query_emb is not None:
        results = search_similar_chunks(query_emb, top_k=TOP_K, chunks=chunks)
    else:
        results = keyword_search(question, top_k=TOP_K, chunks=chunks)

    if not results:
        return jsonify({"answer": "Nao encontrei informacoes relevantes no material. Tente reformular ou adicione material sobre o tema.", "sources": [], "time": 0, "provider": "none"})

    context_parts = []
    for i, c in enumerate(results, 1):
        context_parts.append("[Trecho " + str(i) + " - " + c['book'] + ", p." + str(c['page']) + "]\n" + c['text'])
    context = "\n\n---\n\n".join(context_parts)[:4000]
    system_prompt = SYSTEM_PROMPT + context[:1500]

    messages = []
    if history:
        for msg in history[-MAX_HISTORY:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    response = api_manager.generate(system_prompt, messages)
    elapsed = time.time() - start

    sources = []
    seen = set()
    for r in results:
        key = r['book'] + "-p" + str(r['page'])
        if key not in seen:
            sources.append({"book": r["book"], "page": r["page"], "relevance": round(r["similarity"] * 100, 1)})
            seen.add(key)

    return jsonify({"answer": response["text"], "sources": sources, "time": round(elapsed, 2), "provider": response["provider"], "fallback": response.get("fallback", False)})


@app.route("/api/flashcards", methods=["POST"])
def flashcards():
    data = request.json
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Tema vazio"}), 400
    start = time.time()
    context, sources = get_context_for_topic(topic, top_k=4)
    if not context:
        return jsonify({"error": "Tema nao encontrado no material.", "cards": []})
    system_prompt = FLASHCARD_PROMPT + context[:2000]
    response = api_manager.generate(system_prompt, [{"role": "user", "content": "Gere 5 flashcards sobre: " + topic}])
    cards = parse_json_response(response["text"]) or []
    return jsonify({"cards": cards, "sources": sources, "time": round(time.time() - start, 2), "provider": response["provider"]})


@app.route("/api/quiz", methods=["POST"])
def quiz():
    data = request.json
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Tema vazio"}), 400
    start = time.time()
    context, sources = get_context_for_topic(topic, top_k=4)
    if not context:
        return jsonify({"error": "Tema nao encontrado.", "questions": []})
    system_prompt = QUIZ_PROMPT + context[:2000]
    response = api_manager.generate(system_prompt, [{"role": "user", "content": "Gere 5 questoes sobre: " + topic}])
    questions = parse_json_response(response["text"]) or []
    return jsonify({"questions": questions, "sources": sources, "time": round(time.time() - start, 2), "provider": response["provider"]})


@app.route("/api/mindmap", methods=["POST"])
def mindmap():
    data = request.json
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Tema vazio"}), 400
    start = time.time()
    context, sources = get_context_for_topic(topic, top_k=4)
    if not context:
        return jsonify({"error": "Tema nao encontrado.", "mindmap": None})
    system_prompt = MINDMAP_PROMPT + context[:2000]
    response = api_manager.generate(system_prompt, [{"role": "user", "content": "Gere mapa mental sobre: " + topic}])
    mm = parse_json_response(response["text"])
    return jsonify({"mindmap": mm, "sources": sources, "time": round(time.time() - start, 2), "provider": response["provider"]})


@app.route("/api/exam", methods=["POST"])
def exam():
    data = request.json
    topic = data.get("topic", "").strip()
    types = data.get("types", ["mc"])
    difficulty = data.get("difficulty", "medio")
    n_per_type = data.get("n_per_type", 3)
    weight = data.get("weight", 1)

    if not topic:
        return jsonify({"error": "Tema vazio"}), 400

    start = time.time()
    context, sources = get_context_for_topic(topic, top_k=6)
    if not context:
        return jsonify({"error": "Tema nao encontrado.", "questions": []})

    all_questions = []
    prompt_map = {"mc": EXAM_MC_PROMPT, "disc": EXAM_DISC_PROMPT, "match": EXAM_MATCH_PROMPT, "order": EXAM_ORDER_PROMPT}

    for qtype in types:
        prompt_tmpl = prompt_map.get(qtype)
        if not prompt_tmpl:
            continue
        prompt = prompt_tmpl.format(n=n_per_type, nivel=difficulty, tema=topic, peso=weight)
        system_prompt = prompt + context[:2000]
        response = api_manager.generate(system_prompt, [{"role": "user", "content": "Gere as questoes."}])
        parsed = parse_json_response(response["text"])
        if parsed and isinstance(parsed, list):
            for q in parsed:
                q["type"] = qtype
            all_questions.extend(parsed)

    return jsonify({
        "questions": all_questions,
        "sources": sources,
        "time": round(time.time() - start, 2),
        "topic": topic,
        "difficulty": difficulty
    })


@app.route("/api/study-guide", methods=["POST"])
def study_guide():
    data = request.json
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Tema vazio"}), 400
    start = time.time()
    context, sources = get_context_for_topic(topic, top_k=6)
    if not context:
        return jsonify({"error": "Tema nao encontrado."})
    system_prompt = STUDY_GUIDE_PROMPT + context[:2500]
    response = api_manager.generate(system_prompt, [{"role": "user", "content": "Gere guia de estudo sobre: " + topic}])
    guide = parse_json_response(response["text"])
    return jsonify({"guide": guide, "sources": sources, "time": round(time.time() - start, 2), "provider": response["provider"]})


@app.route("/api/progress", methods=["POST"])
def save_progress():
    user, token = get_user_from_token(request)
    if not user:
        return jsonify({"error": "Nao autenticado"}), 401
    data = request.json
    result = supabase_request("POST", "progress", {
        "user_id": user["id"],
        "type": data.get("type", "quiz"),
        "topic": data.get("topic", ""),
        "score": data.get("score", 0),
        "max_score": data.get("max_score", 0),
        "difficulty": data.get("difficulty", "medio"),
        "details": data.get("details", {})
    }, token=token)
    if result:
        return jsonify({"ok": True})
    return jsonify({"error": "Erro ao salvar"}), 500


@app.route("/api/progress", methods=["GET"])
def get_progress():
    user, token = get_user_from_token(request)
    if not user:
        return jsonify({"error": "Nao autenticado"}), 401
    data = supabase_request("GET", "progress?user_id=eq." + user["id"] + "&order=created_at.desc&limit=50", token=token)
    return jsonify({"progress": data or []})


@app.route("/api/upload-material", methods=["POST"])
def upload_material():
    user, token = get_user_from_token(request)
    if not user:
        return jsonify({"error": "Nao autenticado"}), 401
    data = request.json
    name = data.get("name", "Material sem nome")
    text = data.get("text", "")
    if not text or len(text) < 50:
        return jsonify({"error": "Texto muito curto (minimo 50 caracteres)"}), 400

    chunk_size = 800
    overlap = 100
    chunks = []
    words = text.split()
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        emb = get_query_embedding(chunk_text[:500])
        embedding_list = emb.tolist() if emb is not None else []
        chunks.append({"text": chunk_text, "book": name, "page": str(len(chunks) + 1), "embedding": embedding_list})
        i += chunk_size - overlap

    result = supabase_request("POST", "materials", {
        "user_id": user["id"],
        "name": name,
        "description": text[:200],
        "chunks": chunks,
        "total_chunks": len(chunks)
    }, token=token)

    if result:
        return jsonify({"ok": True, "id": result[0]["id"] if isinstance(result, list) else "", "chunks": len(chunks)})
    return jsonify({"error": "Erro ao salvar material"}), 500


@app.route("/api/materials", methods=["GET"])
def get_materials():
    user, token = get_user_from_token(request)
    if not user:
        return jsonify({"error": "Nao autenticado"}), 401
    data = supabase_request("GET", "materials?user_id=eq." + user["id"] + "&select=id,name,description,total_chunks,created_at&order=created_at.desc", token=token)
    return jsonify({"materials": data or []})


@app.route("/api/materials/<mat_id>", methods=["DELETE"])
def delete_material(mat_id):
    user, token = get_user_from_token(request)
    if not user:
        return jsonify({"error": "Nao autenticado"}), 401
    supabase_request("DELETE", "materials?id=eq." + mat_id + "&user_id=eq." + user["id"], token=token)
    return jsonify({"ok": True})


@app.route("/api/topics")
def topics():
    if not knowledge_base or not knowledge_base["chunks"]:
        return jsonify({"topics": []})
    topic_counts = Counter()
    for chunk in knowledge_base["chunks"]:
        book = chunk.get("book", "")
        if book:
            topic_counts[book] += 1
    books_info = [{"name": b, "chunks": c} for b, c in topic_counts.most_common()]
    return jsonify({"books": books_info, "total_chunks": knowledge_base.get("total_chunks", 0)})


@app.route("/api/status")
def status():
    api_status = api_manager.get_status()
    kb_status = "empty" if (not knowledge_base or not knowledge_base["chunks"]) else "ready"
    return jsonify({
        "status": kb_status,
        "books": knowledge_base.get("books", []) if knowledge_base else [],
        "total_chunks": knowledge_base.get("total_chunks", 0) if knowledge_base else 0,
        "apis": api_status,
        "supabase": bool(SUPABASE_URL and SUPABASE_KEY)
    })


@app.route("/api/providers")
def providers():
    return jsonify(api_manager.get_status())


load_knowledge_base()

if __name__ == "__main__":
    print("=" * 60)
    print("  estudo.RG — Plataforma de Estudos para Concursos")
    print("=" * 60)
    print("Servidor: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
