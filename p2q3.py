import logging
import re
from fastapi import FastAPI, HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel, Field
import torch
from transformers import MarianMTModel, MarianTokenizer

# Configuro o logger para acompanhar o carregamento do modelo e possíveis erros
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"


# Defino o schema de entrada via JSON
class TranslationRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Texto em inglês a ser traduzido para o alemão via LangChain + HuggingFace.",
        examples=[
            "Automated data pipelines reduce manual errors and improve reproducibility across analytics teams."
        ],
    )
    max_new_tokens: int = Field(
        default=256,
        ge=16,
        le=512,
        description="Quantidade máxima de tokens gerados por bloco na tradução.",
    )


# Defino o formato de saída em JSON
class TranslationResponse(BaseModel):
    source_text: str
    translated_text: str
    source_lang: str = "en"
    target_lang: str = "de"
    model: str


# Inicializo a aplicação FastAPI
app = FastAPI(
    title="P2Q3 - Tradução EN-DE com LangChain e HuggingFace",
    description="API de tradução inglês -> alemão utilizando FastAPI, LangChain e Helsinki-NLP/opus-mt-en-de.",
    version="1.0.0",
)

# Seleciono GPU quando disponível, mantendo CPU como alternativa
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrego tokenizer e modelo uma única vez fora da rota para evitar recarregamento a cada requisição
logger.info("Carregando '%s' no dispositivo '%s'...", MODEL_NAME, device)
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


# Exponho .model e .tokenizer para o HuggingFacePipeline do LangChain sem usar task="translation"
class MarianTranslationRunner:
    def __init__(self, tok, mdl, dev, max_new_tokens: int = 256):
        self.tokenizer = tok
        self.model = mdl
        self.device = dev
        self.max_new_tokens = max_new_tokens

    def __call__(self, input_text, **kwargs):
        if isinstance(input_text, list):
            texts = [str(t) for t in input_text]
        else:
            texts = [str(input_text)]

        max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **encoded,
                max_new_tokens=max_tokens,
            )

        results = []
        for seq in generated_ids:
            decoded = self.tokenizer.decode(seq, skip_special_tokens=True).strip()
            results.append({"generated_text": decoded})

        return results if isinstance(input_text, list) else results[0]["generated_text"]

    @property
    def task(self) -> str:
        return "text-generation"


max_model_tokens = getattr(tokenizer, "model_max_length", 512)
if not isinstance(max_model_tokens, int) or max_model_tokens <= 0 or max_model_tokens > 2048:
    max_model_tokens = 512
CHUNK_TOKEN_LIMIT = max(64, max_model_tokens - 32)


def split_into_chunks(text: str, max_tokens: int) -> list[str]:
    """Divide textos longos em blocos respeitando o limite de tokens do modelo."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        sentences = [text.strip()]

    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        candidate = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        if len(tokenizer.encode(candidate, truncation=False)) <= max_tokens:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            if len(tokenizer.encode(sentence, truncation=False)) > max_tokens:
                words = sentence.split()
                word_buffer = ""
                for word in words:
                    word_candidate = f"{word_buffer} {word}".strip() if word_buffer else word
                    if len(tokenizer.encode(word_candidate, truncation=False)) <= max_tokens:
                        word_buffer = word_candidate
                    else:
                        if word_buffer:
                            chunks.append(word_buffer)
                        word_buffer = word
                if word_buffer:
                    current_chunk = word_buffer
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# Monto a cadeia LangChain utilizando PromptTemplate, HuggingFacePipeline e StrOutputParser
prompt = PromptTemplate.from_template("{text}")


def build_translation_chain(max_new_tokens: int):
    """Constrói a cadeia LCEL com HuggingFacePipeline usando o runner MarianMT."""
    runner = MarianTranslationRunner(
        tok=tokenizer,
        mdl=model,
        dev=device,
        max_new_tokens=max_new_tokens,
    )
    llm = HuggingFacePipeline(pipeline=runner)
    return prompt | llm | StrOutputParser()


@app.post("/translate", response_model=TranslationResponse)
def translate_en_to_de(payload: TranslationRequest) -> TranslationResponse:
    """Recebe um texto em inglês via POST e retorna a tradução em alemão usando LangChain."""
    try:
        chunks = split_into_chunks(payload.text, max_tokens=CHUNK_TOKEN_LIMIT)
        chain = build_translation_chain(payload.max_new_tokens)

        translated_parts: list[str] = []
        for chunk in chunks:
            result = chain.invoke({"text": chunk})
            if isinstance(result, list):
                text_out = " ".join(
                    str(item.get("generated_text", item) if isinstance(item, dict) else item).strip()
                    for item in result
                ).strip()
            else:
                text_out = str(result).strip()
            if text_out:
                translated_parts.append(text_out)

        translated_text = " ".join(translated_parts).strip()

        return TranslationResponse(
            source_text=payload.text,
            translated_text=translated_text,
            model=MODEL_NAME,
        )

    except Exception as exc:
        logger.error("Erro na tradução EN->DE via LangChain: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Falha interna ao traduzir o texto para alemão com opus-mt-en-de.",
        ) from exc