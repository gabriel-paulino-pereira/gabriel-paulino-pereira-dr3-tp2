import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Configuro o logger básico para acompanhar o carregamento e possíveis falhas na API
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "gpt2"

# Defino o contrato de entrada esperado no corpo da requisição HTTP (JSON)
class TextGenerationRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        description="Frase ou trecho inicial para o GPT-2 gerar uma continuação.",
        examples=["Artificial intelligence in public data analysis is"],
    )
    max_new_tokens: int = Field(
        default=60,
        ge=1,
        le=200,
        description="Quantidade máxima de novos tokens gerados após o prompt.",
    )
    temperature: float = Field(
        default=0.8,
        gt=0.0,
        le=2.0,
        description="Controla a aleatoriedade da amostragem (valores menores deixam mais determinístico).",
    )

# Defino o formato padronizado da resposta JSON
class TextGenerationResponse(BaseModel):
    prompt: str
    generated_text: str
    model: str

# Inicializo a aplicação FastAPI
app = FastAPI(
    title="P1Q1 - Geração de Texto com GPT-2",
    description="API em FastAPI utilizando transformers e o modelo GPT-2 para continuação de texto.",
    version="1.0.0",
)

# Seleciono GPU se disponível, mantendo CPU como fallback padrão
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrego tokenizer e modelo uma única vez na inicialização do módulo para evitar latência repetida
logger.info("Carregando tokenizer e modelo '%s' no dispositivo '%s'...", MODEL_NAME, device)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)

# Garanto um token de padding definido para o GPT-2 (que nativamente não possui pad_token)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Coloco o modelo em modo de avaliação para desativar camadas como dropout
model.eval()


@app.post("/generate", response_model=TextGenerationResponse)
def generate_text(payload: TextGenerationRequest) -> TextGenerationResponse:
    """Recebe um prompt via POST e retorna a continuação gerada pelo GPT-2."""
    try:
        # Converto o texto de entrada em tensores de IDs de tokens e attention_mask
        inputs = tokenizer(
            payload.prompt,
            return_tensors="pt",
            padding=True,
        ).to(device)

        # Desativo o cálculo de gradientes para economizar memória e acelerar a inferência
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=payload.max_new_tokens,
                temperature=payload.temperature,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decodifico apenas os tokens gerados após o tamanho do prompt original para obter a continuação
        prompt_token_len = inputs["input_ids"].shape[1]
        continuation_ids = output_ids[0][prompt_token_len:]
        continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()

        # Monto o texto completo combinando a entrada original com a continuação gerada
        full_text = f"{payload.prompt.strip()} {continuation}".strip()

        return TextGenerationResponse(
            prompt=payload.prompt,
            generated_text=full_text,
            model=MODEL_NAME,
        )

    except Exception as exc:
        logger.error("Erro durante a geração de texto: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Falha interna ao processar a geração de texto com o modelo GPT-2.",
        ) from exc