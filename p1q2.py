import logging
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import MarianMTModel, MarianTokenizer

# Configuro o logger para acompanhar o carregamento do modelo e possíveis falhas na API
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"


# Defino o schema de entrada da requisição HTTP
class TranslationRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Texto em inglês (frase curta ou parágrafo mais longo) a ser traduzido para o francês.",
        examples=[
            "Automated data pipelines reduce manual errors and improve reproducibility across analytics teams."
        ],
    )
    max_new_tokens: int = Field(
        default=256,
        ge=16,
        le=512,
        description="Limite máximo de tokens gerados por bloco na tradução.",
    )


# Defino o formato de saída em JSON
class TranslationResponse(BaseModel):
    source_text: str
    translated_text: str
    source_lang: str = "en"
    target_lang: str = "fr"
    model: str


# Inicializo a aplicação FastAPI
app = FastAPI(
    title="P1Q2 - Tradução EN-FR com opus-mt-en-fr",
    description="API em FastAPI utilizando transformers e o modelo Helsinki-NLP/opus-mt-en-fr.",
    version="1.0.0",
)

# Seleciono GPU quando disponível, caso contrário uso CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrego tokenizer e modelo uma única vez na subida da aplicação
logger.info("Carregando modelo de tradução '%s' no dispositivo '%s'...", MODEL_NAME, device)
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Respeito a janela máxima de tokens do modelo evitando estouro em textos longos
max_model_tokens = getattr(tokenizer, "model_max_length", 512)
if not isinstance(max_model_tokens, int) or max_model_tokens <= 0 or max_model_tokens > 2048:
    max_model_tokens = 512
CHUNK_TOKEN_LIMIT = max(64, max_model_tokens - 32)


def split_into_chunks(text: str, max_tokens: int) -> list[str]:
    """Divide textos longos em blocos menores por sentenças/palavras respeitando o limite de tokens."""
    # Separo o texto em sentenças mantendo a pontuação final
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        sentences = [text.strip()]

    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        candidate = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        token_count = len(tokenizer.encode(candidate, truncation=False))

        if token_count <= max_tokens:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # Se uma única sentença já exceder o limite, quebro por palavras
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


@app.post("/translate", response_model=TranslationResponse)
def translate_text(payload: TranslationRequest) -> TranslationResponse:
    """Recebe um texto em inglês via POST e retorna a tradução em francês."""
    try:
        chunks = split_into_chunks(payload.text, max_tokens=CHUNK_TOKEN_LIMIT)
        translated_parts: list[str] = []

        # Tokenizo todos os blocos em lote para melhorar o aproveitamento da inferência
        encoded = tokenizer(
            chunks,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_model_tokens,
        ).to(device)

        # Desativo cálculo de gradientes durante a inferência
        with torch.no_grad():
            generated_ids = model.generate(
                **encoded,
                max_new_tokens=payload.max_new_tokens,
            )

        for seq in generated_ids:
            decoded = tokenizer.decode(seq, skip_special_tokens=True).strip()
            if decoded:
                translated_parts.append(decoded)

        translated_text = " ".join(translated_parts).strip()

        return TranslationResponse(
            source_text=payload.text,
            translated_text=translated_text,
            model=MODEL_NAME,
        )

    except Exception as exc:
        logger.error("Erro na tradução EN->FR: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Falha interna ao traduzir o texto com o modelo opus-mt-en-fr.",
        ) from exc