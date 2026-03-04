"""
gerar_kb_concurso.py — estudo.RG Concursos
===========================================
Coloque todos os PDFs na pasta 'pdfs_concurso' e rode:
    python gerar_kb_concurso.py

Vai gerar o arquivo data/kb_concurso.json pronto para subir no GitHub.

Dependencias:
    pip install pymupdf python-dotenv requests numpy
"""

import os
import json
import time
import re
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# =============================================
# CONFIG
# =============================================
PASTA_PDFS      = "pdfs_concurso"           # pasta com seus PDFs
SAIDA_KB        = "data/kb_concurso.json"   # arquivo gerado
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
CHUNK_SIZE      = 600    # palavras por chunk
CHUNK_OVERLAP   = 80     # sobreposição entre chunks
DELAY_EMBEDDING = 12.0    # segundos entre chamadas (evita rate limit)


# =============================================
# EXTRAÇÃO DE TEXTO DO PDF
# =============================================
def extrair_texto_pdf(caminho):
    try:
        import fitz  # pymupdf
        doc = fitz.open(caminho)
        paginas = []
        for num, pagina in enumerate(doc, 1):
            texto = pagina.get_text("text")
            texto = limpar_texto(texto)
            if len(texto.strip()) > 50:
                paginas.append({"num": num, "texto": texto})
        doc.close()
        return paginas
    except Exception as e:
        print(f"   ERRO ao ler {caminho}: {e}")
        return []


def limpar_texto(texto):
    # Remove hifenização no fim de linha
    texto = re.sub(r'-\n', '', texto)
    # Junta linhas quebradas (mantém parágrafos)
    texto = re.sub(r'(?<!\n)\n(?!\n)', ' ', texto)
    # Remove espaços duplos
    texto = re.sub(r' {2,}', ' ', texto)
    # Remove caracteres estranhos
    texto = re.sub(r'[^\w\s\.,;:!?()\/\-–—""\'\"àáâãäéêëíîïóôõöúûüçñÀÁÂÃÄÉÊËÍÎÏÓÔÕÖÚÛÜÇÑ°º§]', ' ', texto)
    return texto.strip()


# =============================================
# CHUNKING
# =============================================
def criar_chunks(paginas, nome_livro):
    """Divide o texto em chunks com sobreposição."""
    chunks = []

    # Junta tudo preservando informação de página
    tokens_com_pagina = []
    for p in paginas:
        palavras = p["texto"].split()
        for palavra in palavras:
            tokens_com_pagina.append((palavra, p["num"]))

    total = len(tokens_com_pagina)
    i = 0
    while i < total:
        janela = tokens_com_pagina[i:i + CHUNK_SIZE]
        texto_chunk = " ".join(w for w, _ in janela)
        # Página predominante no chunk
        paginas_chunk = [pg for _, pg in janela]
        pagina_principal = max(set(paginas_chunk), key=paginas_chunk.count)

        if len(texto_chunk.strip()) > 100:
            chunks.append({
                "text": texto_chunk,
                "book": nome_livro,
                "page": str(pagina_principal),
                "embedding": []  # preenchido depois
            })

        i += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# =============================================
# EMBEDDINGS VIA GEMINI
# =============================================
def gerar_embedding(texto):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "cole_sua_chave_aqui":
        return None
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={GEMINI_API_KEY}"
        payload = {
            "model": "models/gemini-embedding-001",
            "content": {"parts": [{"text": texto[:2000]}]}
        }
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            return r.json()["embedding"]["values"]
        elif r.status_code == 429:
            print("   Rate limit — aguardando 10s...")
            time.sleep(30)
            r2 = requests.post(url, json=payload, timeout=15)
            if r2.status_code == 200:
                return r2.json()["embedding"]["values"]
    except Exception as e:
        print(f"   Erro embedding: {e}")
    return None


# =============================================
# MAIN
# =============================================
def main():
    print("=" * 55)
    print("  estudo.RG — Gerador de Base de Conhecimento")
    print("=" * 55)

    # Verifica pasta de PDFs
    if not os.path.exists(PASTA_PDFS):
        os.makedirs(PASTA_PDFS)
        print(f"\nPasta '{PASTA_PDFS}' criada.")
        print(f"Coloque seus PDFs lá e rode novamente.\n")
        return

    # Busca PDFs recursivamente em subpastas
    pdfs = []  # lista de (caminho_completo, nome_livro, materia)
    for raiz, dirs, arquivos in os.walk(PASTA_PDFS):
        dirs.sort()
        for arq in sorted(arquivos):
            if arq.lower().endswith(".pdf"):
                caminho_completo = os.path.join(raiz, arq)
                # Matéria = nome da subpasta (ou "Geral" se na raiz)
                materia = os.path.basename(raiz)
                if materia == PASTA_PDFS:
                    materia = "Geral"
                nome_livro = f"[{materia}] {os.path.splitext(arq)[0]}"
                pdfs.append((caminho_completo, nome_livro, materia))

    if not pdfs:
        print(f"\nNenhum PDF encontrado em '{PASTA_PDFS}' ou subpastas.")
        print("Coloque seus PDFs lá e rode novamente.\n")
        return

    # Agrupa por matéria para exibição
    materias = {}
    for _, _, mat in pdfs:
        materias[mat] = materias.get(mat, 0) + 1

    print(f"\n{len(pdfs)} PDF(s) em {len(materias)} matéria(s):")
    for mat, qtd in sorted(materias.items()):
        print(f"  📁 {mat}: {qtd} PDF(s)")

    if False and GEMINI_API_KEY != "cole_sua_chave_aqui":
        print(f"\nEmbeddings: Gemini API ✅")
    else:
        print(f"\nEmbeddings: DESATIVADO (sem GEMINI_API_KEY)")
        print("  O sistema usará busca por palavras-chave no lugar.")

    print("\nIniciando processamento...\n")

    todos_chunks = []
    todos_livros = []

    for caminho, nome_livro, materia in pdfs:
        print(f"📄 {nome_livro}")

        # Extrai texto
        paginas = extrair_texto_pdf(caminho)
        if not paginas:
            print(f"   ⚠ Nenhum texto extraído — PDF pode ser imagem (scaneado)\n")
            continue

        total_palavras = sum(len(p["texto"].split()) for p in paginas)
        print(f"   {len(paginas)} páginas | ~{total_palavras:,} palavras")

        # Cria chunks
        chunks = criar_chunks(paginas, nome_livro)
        print(f"   {len(chunks)} chunks criados")

        # Gera embeddings
        if False and GEMINI_API_KEY != "cole_sua_chave_aqui":
            print(f"   Gerando embeddings...", end="", flush=True)
            ok = 0
            for j, chunk in enumerate(chunks):
                emb = gerar_embedding(chunk["text"])
                if emb:
                    chunk["embedding"] = emb
                    ok += 1
                else:
                    chunk["embedding"] = []
                if (j + 1) % 10 == 0:
                    print(f" {j+1}", end="", flush=True)
                time.sleep(DELAY_EMBEDDING)
            print(f" ✅ {ok}/{len(chunks)} embeddings")
        else:
            print(f"   Embeddings pulados (sem chave)")

        todos_chunks.extend(chunks)
        todos_livros.append(nome_livro)
        print()

    if not todos_chunks:
        print("Nenhum chunk gerado. Verifique os PDFs.\n")
        return

    # Monta e salva o JSON
    os.makedirs("data", exist_ok=True)
    kb = {
        "chunks": todos_chunks,
        "books": todos_livros,
        "total_chunks": len(todos_chunks)
    }

    with open(SAIDA_KB, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

    tamanho_mb = os.path.getsize(SAIDA_KB) / 1024 / 1024

    print("=" * 55)
    print(f"  ✅ Base gerada com sucesso!")
    print(f"  Arquivo : {SAIDA_KB}")
    print(f"  Chunks  : {len(todos_chunks)}")
    print(f"  Livros  : {len(todos_livros)}")
    print(f"  Tamanho : {tamanho_mb:.1f} MB")
    print("=" * 55)
    print()
    print("Próximo passo:")
    print("  Copie o arquivo data/kb_concurso.json para o")
    print("  seu repositório e faça push para o GitHub.")
    print("  O Render vai fazer o redeploy automaticamente.")
    print()

    if tamanho_mb > 50:
        print("⚠ Arquivo grande (>50MB). Considere dividir os PDFs")
        print("  em lotes menores para não travar o Render free tier.")
        print()


if __name__ == "__main__":
    main()
