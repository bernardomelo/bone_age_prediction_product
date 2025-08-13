from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import psutil
from datetime import datetime
import numpy as np
from PIL import Image
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Criar diretório para salvar imagens recebidas
# para testes
# IMAGES_DIR = Path("image-requests")
# IMAGES_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Bone Age Prediction API",
    description="MVP para predição de maturidade óssea a partir de imagens",
    version="1.0.0"
)

# CORS para desenvolvimento
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Para testes
# def save_received_image(file_contents, original_filename):
#     """
#     Salva imagem recebida no diretório image-requests
#     """
#     try:
#         # Gerar nome único baseado no timestamp
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microsegundos truncados
#         file_extension = Path(original_filename).suffix.lower()
#         if not file_extension:
#             file_extension = '.jpg'  # default
#
#         saved_filename = f"{timestamp}_{original_filename}"
#         file_path = IMAGES_DIR / saved_filename
#
#         # Salvar arquivo
#         with open(file_path, 'wb') as f:
#             f.write(file_contents)
#
#         logger.info(f"Imagem salva em: {file_path}")
#         return str(file_path), saved_filename
#
#     except Exception as e:
#         logger.error(f"Erro ao salvar imagem: {e}")
#         return None, None


def validate_image_file(file: UploadFile) -> dict:
    """
    Validação de arquivo de imagem
    """
    result = {"is_valid": False, "error": None}

    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            result["error"] = "Arquivo deve ser uma imagem (JPEG, PNG, etc.)"
            return result

        if file.size > 20 * 1024 * 1024:  # 20MB max
            result["error"] = "Arquivo muito grande (máximo 20MB)"
            return result

        result["is_valid"] = True
        return result

    except Exception as e:
        result["error"] = f"Erro na validação: {str(e)}"
        return result


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint principal: predição de idade óssea a partir de imagem
    Aceita arquivos de imagem (JPEG, PNG, etc.)
    """
    start_time = time.time()

    try:
        validation = validate_image_file(file)
        if not validation["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail=validation["error"]
            )

        contents = await file.read()
        logger.info(f"Imagem recebida: {file.filename} ({len(contents)} bytes)")

        # Salvar imagem recebida para teste
        # saved_path, saved_filename = save_received_image(contents, file.filename)
        # if not saved_path:
        #     logger.warning("Falha ao salvar imagem, mas continuando processamento...")

        # Usando mock por enquanto
        # result = bone_age_model.predict(processed_array)
        result = {"status":"success",
                  "filename":"output_image7.png",
                  "prediction":{"predicted_age_months":92.7,"predicted_age_years":7.7,"confidence":0.894,"model_status":"MOCK - modelo real será carregado depois","array_shape":[1,384,384,3],"processing_time_ms":282.56},
                  "preprocessing_info":{"shape":[1,384,384,3], "dtype":"float32","min_value":-123.68000030517578,"max_value":127.06099700927734,"mean_value":-74.85670471191406},
                  "timestamp":"2025-08-13T06:08:23.956459"}

        logger.info(f"Mocked predicted bone age: {result}")

        processing_time = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Tempo de processamento para {file.filename}: {processing_time}")
        result["processing_time_ms"] = processing_time

        response = {
            "status": "success",
            "filename": file.filename,
            # "saved_as": saved_filename,
            # "saved_path": saved_path,
            "prediction": result,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Predição concluída em {processing_time}ms")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro interno: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno do servidor: {str(e)}"
        )


@app.get("/")
def root():
    """
    Endpoint raiz com informações básicas
    """
    return {
        "message": "Bone Age Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict - POST - Predição de idade óssea (imagens JPEG/PNG)",
            "health": "/health - GET - Status do sistema",
            "docs": "/docs - Documentação interativa"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)