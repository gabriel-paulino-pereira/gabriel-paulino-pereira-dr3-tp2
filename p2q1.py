import logging
from fastapi import FastAPI, HTTPException
from langchain_community.llms import FakeListLLM
from pydantic import BaseModel, Field

# Configuro o logger para acompanhar o funcionamento do protótipo
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Defino o contrato de entrada via JSON para o chatbot
class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="Mensagem ou pergunta enviada pelo usuário ao chatbot.",
        examples=["O que e FastAPI?"],
    )

# Defino o formato da resposta retornada pela API
class ChatResponse(BaseModel):
    user_message: str
    bot_reply: str
    llm_type: str = "FakeListLLM"

# Inicializo a aplicação FastAPI
app = FastAPI(
    title="P2Q1 - Chatbot Simples com Fake LLM (LangChain)",
    description="Protótipo de chatbot utilizando FastAPI e FakeListLLM do LangChain.",
    version="1.0.0",
)

# Defino respostas simuladas pré-definidas para o FakeListLLM
PREDEFINED_RESPONSES: list[str] = [
    "FastAPI e um framework web moderno e rapido em Python baseado em tipagem com Pydantic.",
    "LangChain ajuda a orquestrar fluxos, prompts e modelos de linguagem de forma modular.",
    "FakeListLLM permite testar rotas, contratos JSON e integracoes sem gastar tokens ou depender de rede.",
    "Resposta padrao simulada: fluxo de chatbot validado com sucesso.",
]

# Mapeamento simples de perguntas-chave para respostas pré-definidas conhecidas
KEYWORD_MAP: dict[str, str] = {
    "fastapi": PREDEFINED_RESPONSES[0],
    "langchain": PREDEFINED_RESPONSES[1],
    "fake": PREDEFINED_RESPONSES[2],
    "teste": PREDEFINED_RESPONSES[2],
}

# Instancio o Fake LLM com a lista de respostas pré-definidas
fake_llm = FakeListLLM(responses=PREDEFINED_RESPONSES)


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Recebe uma mensagem via POST e retorna uma resposta simulada via FakeListLLM."""
    try:
        normalized = payload.message.lower()

        # Verifico se há correspondência direta com alguma pergunta/tema pré-definido
        matched_response = None
        for keyword, predefined in KEYWORD_MAP.items():
            if keyword in normalized:
                matched_response = predefined
                break

        # Invoco o FakeListLLM do LangChain (se houver match, reordeno para retornar a resposta correspondente)
        if matched_response is not None:
            llm_para_consulta = FakeListLLM(responses=[matched_response])
            reply = llm_para_consulta.invoke(payload.message)
        else:
            reply = fake_llm.invoke(payload.message)

        return ChatResponse(
            user_message=payload.message,
            bot_reply=str(reply),
        )

    except Exception as exc:
        logger.error("Erro ao processar mensagem no Fake LLM: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Falha interna ao simular resposta com Fake LLM.",
        ) from exc