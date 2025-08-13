from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import psutil
from datetime import datetime
import numpy as np
from PIL import Image
import logging

# Imports internos
from utils.image_pre_processing import ImagePreprocessor
#from utils.model_handler import BoneAgeModel


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

# try:
#     # Inicializar Modelo
#     bone_age_model = BoneAgeModel(model_path='attentionv3.h5')
#     logger.info(f"- Model carregado com sucesso")
# except Exception as e:
#     raise HTTPException(
#         status_code=400,
#         detail=f"Erro no carregamento do modelo: {str(e)}"
#     )

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

def mock_predict_bone_age(processed_array) -> dict:
    """
    MOCK de predição - substituir pelo modelo real
    """
    # Simulando processamento do modelo
    time.sleep(0.2)

    # Mock simples: gerar idade aleatória
    predicted_age = np.random.uniform(6.0, 18.0)
    confidence = np.random.uniform(0.85, 0.98)

    return {
        "predicted_age_months": round(predicted_age * 12, 1),
        "predicted_age_years": round(predicted_age, 1),
        "confidence": round(confidence, 3),
        "model_status": "MOCK - modelo real será carregado depois",
        "array_shape": list(processed_array.shape)
    }


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

    preprocessor = ImagePreprocessor(target_size=(384, 384))

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

        try:
            processed_array = preprocessor.preprocess_from_bytes(contents)
            logger.info(f"Imagem pré-processada com sucesso - Shape: {processed_array.shape}")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Erro no pré-processamento da imagem: {str(e)}"
            )

        try:
            # Usando mock por enquanto
            # result = bone_age_model.predict(processed_array)
            result = mock_predict_bone_age(processed_array)
            logger.info(f"Mocked predicted bone age: {result}")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Erro na inferência do modelo: {str(e)}"
            )

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

        logger.info(f"Predição mock concluída em {processing_time}ms")
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