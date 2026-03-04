# multi_api_concurso.py — estudo.RG Concursos
import os
import requests
import time
import random


class MultiAPIManager:
    def __init__(self):
        self.providers = [
            {
                "name": "Gemini",
                "key_env": "GEMINI_API_KEY",
                "available": False,
                "errors": 0,
                "last_used": 0,
            },
            {
                "name": "Groq",
                "key_env": "GROQ_API_KEY",
                "available": False,
                "errors": 0,
                "last_used": 0,
            },
            {
                "name": "Mistral",
                "key_env": "MISTRAL_API_KEY",
                "available": False,
                "errors": 0,
                "last_used": 0,
            },
            {
                "name": "Cohere",
                "key_env": "COHERE_API_KEY",
                "available": False,
                "errors": 0,
                "last_used": 0,
            },
            {
                "name": "HuggingFace",
                "key_env": "HF_API_KEY",
                "available": False,
                "errors": 0,
                "last_used": 0,
            },
            {
                "name": "Together",
                "key_env": "TOGETHER_API_KEY",
                "available": False,
                "errors": 0,
                "last_used": 0,
            },
            {
                "name": "OpenRouter",
                "key_env": "OPENROUTER_API_KEY",
                "available": False,
                "errors": 0,
                "last_used": 0,
            },
        ]
        self._init_providers()

    def _init_providers(self):
        for p in self.providers:
            key = os.getenv(p["key_env"], "")
            if key and key not in ("", "cole_sua_chave_aqui", "sua_chave_aqui"):
                p["available"] = True
                p["key"] = key
            else:
                p["key"] = ""

    def _call_gemini(self, key, system_prompt, messages, max_tokens=2048):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7},
        }
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text

    def _call_groq(self, key, system_prompt, messages, max_tokens=2048):
        url = "https://api.groq.com/openai/v1/chat/completions"
        msgs = [{"role": "system", "content": system_prompt}] + messages
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _call_mistral(self, key, system_prompt, messages, max_tokens=2048):
        url = "https://api.mistral.ai/v1/chat/completions"
        msgs = [{"role": "system", "content": system_prompt}] + messages
        payload = {
            "model": "mistral-small-latest",
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _call_cohere(self, key, system_prompt, messages, max_tokens=2048):
        url = "https://api.cohere.com/v2/chat"
        msgs = []
        for m in messages:
            msgs.append({"role": m["role"], "content": m["content"]})
        payload = {
            "model": "command-r-plus",
            "system_prompt": system_prompt,
            "messages": msgs,
            "max_tokens": max_tokens,
        }
        r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"][0]["text"]

    def _call_huggingface(self, key, system_prompt, messages, max_tokens=2048):
        url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1/v1/chat/completions"
        msgs = [{"role": "system", "content": system_prompt}] + messages
        payload = {
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload, timeout=45)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _call_together(self, key, system_prompt, messages, max_tokens=2048):
        url = "https://api.together.xyz/v1/chat/completions"
        msgs = [{"role": "system", "content": system_prompt}] + messages
        payload = {
            "model": "meta-llama/Llama-3-70b-chat-hf",
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _call_openrouter(self, key, system_prompt, messages, max_tokens=2048):
        url = "https://openrouter.ai/api/v1/chat/completions"
        msgs = [{"role": "system", "content": system_prompt}] + messages
        payload = {
            "model": "meta-llama/llama-3.1-8b-instruct:free",
            "messages": msgs,
            "max_tokens": max_tokens,
        }
        r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "HTTP-Referer": "https://estudo.rg", "X-Title": "estudo.RG"}, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def generate(self, system_prompt, messages, max_tokens=2048):
        available = [p for p in self.providers if p["available"] and p["errors"] < 3]

        if not available:
            # Reset error counts e tenta de novo
            for p in self.providers:
                if p["available"]:
                    p["errors"] = 0
            available = [p for p in self.providers if p["available"]]

        if not available:
            return {
                "text": "Nenhuma API configurada. Adicione pelo menos uma chave no arquivo .env.",
                "provider": "none",
                "fallback": False,
            }

        # Ordena por menos erros e menos usado recentemente
        available.sort(key=lambda x: (x["errors"], x["last_used"]))

        for provider in available:
            try:
                name = provider["name"]
                key = provider["key"]
                start = time.time()

                if name == "Gemini":
                    text = self._call_gemini(key, system_prompt, messages, max_tokens)
                elif name == "Groq":
                    text = self._call_groq(key, system_prompt, messages, max_tokens)
                elif name == "Mistral":
                    text = self._call_mistral(key, system_prompt, messages, max_tokens)
                elif name == "Cohere":
                    text = self._call_cohere(key, system_prompt, messages, max_tokens)
                elif name == "HuggingFace":
                    text = self._call_huggingface(key, system_prompt, messages, max_tokens)
                elif name == "Together":
                    text = self._call_together(key, system_prompt, messages, max_tokens)
                elif name == "OpenRouter":
                    text = self._call_openrouter(key, system_prompt, messages, max_tokens)
                else:
                    continue

                provider["errors"] = max(0, provider["errors"] - 1)
                provider["last_used"] = time.time()
                elapsed = round(time.time() - start, 2)
                print(f"[MultiAPI] {name} respondeu em {elapsed}s")
                return {"text": text, "provider": name, "fallback": name != available[0]["name"], "time": elapsed}

            except Exception as e:
                provider["errors"] += 1
                print(f"[MultiAPI] Erro em {provider['name']}: {e}")
                continue

        return {
            "text": "Todas as APIs falharam. Verifique suas chaves de API e conexão.",
            "provider": "none",
            "fallback": True,
        }

    def get_status(self):
        result = []
        for p in self.providers:
            result.append({
                "name": p["name"],
                "available": p["available"],
                "errors": p["errors"],
                "configured": bool(p.get("key")),
            })
        return result
