import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import httpx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Carrego variáveis de ambiente do .env 
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

# Lista inicial de modelos candidatos (pode ser sobrescrita via LLM_MODEL no .env)
GROQ_CANDIDATE_MODELS = [
    m.strip()
    for m in [
        os.getenv("LLM_MODEL", ""),
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
    ]
    if m and m.strip()
]

OPENAI_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()


class TranslationRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Texto em inglês a ser traduzido para o francês via LangChain.",
        examples=[
            "Automated pipelines and clear data contracts reduce errors across analytics workflows."
        ],
    )


class TranslationResponse(BaseModel):
    source_text: str
    translated_text: str
    source_lang: str = "en"
    target_lang: str = "fr"
    provider: str
    model: str


app = FastAPI(
    title="P2Q2 - Tradução EN-FR com LangChain (Groq / OpenAI)",
    description="API de tradução inglês -> francês utilizando FastAPI e LangChain com Groq ou OpenAI.",
    version="1.0.0",
)

translation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise translator. Translate the user's English text into natural, "
            "accurate French. Return ONLY the translated French text without explanations or quotes.",
        ),
        ("human", "{text}"),
    ]
)


def build_chat_model(model_name: str):
    """Instancia o modelo de chat do LangChain de acordo com o provedor configurado."""
    if LLM_PROVIDER == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Defina OPENAI_API_KEY no ambiente ou no arquivo .env.")
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.0)

    if LLM_PROVIDER == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Defina GROQ_API_KEY no ambiente ou no arquivo .env.")
        return ChatGroq(model=model_name, api_key=api_key, temperature=0.0)

    raise ValueError(f"Provedor não suportado: '{LLM_PROVIDER}'. Use 'groq' ou 'openai'.")


def fetch_active_groq_models() -> list[str]:
    """Consulta os modelos atualmente disponíveis na Groq para a chave configurada."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return []
    try:
        resp = httpx.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        # Filtra modelos que não sejam de áudio/whisper/guard quando possível
        models = [
            item.get("id")
            for item in data
            if item.get("id")
            and "whisper" not in item.get("id", "").lower()
            and "guard" not in item.get("id", "").lower()
        ]
        return models
    except Exception as exc:
        logger.warning("Não foi possível listar modelos ativos na Groq: %s", exc)
        return []


def is_model_availability_error(exc: Exception) -> bool:
    """Identifica erros de modelo inexistente, sem acesso ou descontinuado."""
    err = str(exc).lower()
    markers = (
        "model_not_found",
        "model_decommissioned",
        "does not exist",
        "no longer supported",
        "decommissioned",
        "404",
    )
    return any(marker in err for marker in markers)


@app.post("/translate", response_model=TranslationResponse)
def translate_en_to_fr(payload: TranslationRequest) -> TranslationResponse:
    """Recebe texto em inglês via POST e retorna a tradução em francês usando LangChain."""
    models_to_try = [OPENAI_MODEL] if LLM_PROVIDER == "openai" else list(GROQ_CANDIDATE_MODELS)

    if LLM_PROVIDER == "groq":
        dynamic_models = fetch_active_groq_models()
        for m in dynamic_models:
            if m not in models_to_try:
                models_to_try.append(m)

    last_error: Exception | None = None

    for model_name in models_to_try:
        try:
            llm = build_chat_model(model_name)
            chain = translation_prompt | llm | StrOutputParser()
            translated = chain.invoke({"text": payload.text}).strip()
            return TranslationResponse(
                source_text=payload.text,
                translated_text=translated,
                provider=LLM_PROVIDER,
                model=model_name,
            )
        except Exception as exc:
            last_error = exc
            if LLM_PROVIDER == "groq" and is_model_availability_error(exc):
                logger.warning("Modelo '%s' indisponível/descontinuado na Groq; tentando próximo...", model_name)
                continue
            break

    logger.error("Erro na tradução via LangChain: %s", last_error)
    raise HTTPException(
        status_code=502,
        detail=f"Falha ao comunicar com '{LLM_PROVIDER}': {last_error}",
    ) from last_error