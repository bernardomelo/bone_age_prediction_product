import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import io
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Classe para pré-processamento de imagens para predição de idade óssea
    """

    def __init__(self, target_size=(384, 384)):
        self.target_size = target_size
        logger.info(f"ImagePreprocessor inicializado com target_size: {target_size}")

    def load_image_from_bytes(self, image_bytes):
        """
        Carrega imagem a partir de bytes
        """
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            logger.info(f"Imagem carregada - Tamanho original: {pil_image.size}, Modo: {pil_image.mode}")
            return pil_image
        except Exception as e:
            logger.error(f"Erro ao carregar imagem: {e}")
            raise ValueError(f"Não foi possível carregar a imagem: {e}")

    def load_image_from_path(self, image_path):
        """
        Carrega imagem a partir de caminho de arquivo
        """
        try:
            img = image.load_img(image_path, target_size=self.target_size)
            logger.info(f"Imagem carregada de {image_path} - Tamanho: {img.size}")
            return img
        except Exception as e:
            logger.error(f"Erro ao carregar imagem do path: {e}")
            raise ValueError(f"Não foi possível carregar a imagem: {e}")

    def resize_image(self, pil_image):
        """
        Redimensiona imagem PIL para o target_size
        """
        try:
            resized_image = pil_image.resize(self.target_size, Image.Resampling.LANCZOS)
            logger.info(f"Imagem redimensionada para: {resized_image.size}")
            return resized_image
        except Exception as e:
            logger.error(f"Erro ao redimensionar imagem: {e}")
            raise ValueError(f"Erro no redimensionamento: {e}")

    def pil_to_array(self, pil_image):
        """
        Converte PIL Image para numpy array
        """
        try:
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            img_array = image.img_to_array(pil_image)
            logger.info(f"Array criado - Shape: {img_array.shape}")
            return img_array
        except Exception as e:
            logger.error(f"Erro ao converter PIL para array: {e}")
            raise ValueError(f"Erro na conversão: {e}")

    def apply_vgg_preprocessing(self, img_array):
        """
        Aplica pré-processamento do VGG16
        """
        try:
            preprocessed = preprocess_input(img_array.copy())
            logger.info("Pré-processamento VGG16 aplicado")
            return preprocessed
        except Exception as e:
            logger.error(f"Erro no pré-processamento VGG16: {e}")
            raise ValueError(f"Erro no pré-processamento: {e}")

    def add_batch_dimension(self, img_array):
        """
        Adiciona dimensão de batch (para predição)
        """
        try:
            batched = np.expand_dims(img_array, axis=0)
            logger.info(f"Dimensão de batch adicionada - Shape final: {batched.shape}")
            return batched
        except Exception as e:
            logger.error(f"Erro ao adicionar batch dimension: {e}")
            raise ValueError(f"Erro na preparação para predição: {e}")

    def preprocess_from_bytes(self, image_bytes):
        """
        Pipeline completo: bytes -> array pronto para predição
        """
        try:
            pil_image = self.load_image_from_bytes(image_bytes)

            resized_image = self.resize_image(pil_image)

            img_array = self.pil_to_array(resized_image)

            preprocessed = self.apply_vgg_preprocessing(img_array)

            final_array = self.add_batch_dimension(preprocessed)

            logger.info(f"Pré-processamento completo - Shape final: {final_array.shape}")
            return final_array

        except Exception as e:
            logger.error(f"Erro no pipeline completo: {e}")
            raise

    def preprocess_from_path(self, image_path):
        """
        Pipeline completo: path -> array pronto para predição
        """
        try:
            img = self.load_image_from_path(image_path)

            img_array = self.pil_to_array(img)

            preprocessed = self.apply_vgg_preprocessing(img_array)

            final_array = self.add_batch_dimension(preprocessed)

            logger.info(f"Pré-processamento de path completo - Shape: {final_array.shape}")
            return final_array

        except Exception as e:
            logger.error(f"Erro no pipeline de path: {e}")
            raise

    def preprocess_pil_image(self, pil_image):
        """
        Pipeline completo: PIL Image -> array pronto para predição
        """
        try:
            resized_image = self.resize_image(pil_image)

            img_array = self.pil_to_array(resized_image)

            preprocessed = self.apply_vgg_preprocessing(img_array)

            final_array = self.add_batch_dimension(preprocessed)

            logger.info(f"Pré-processamento PIL completo - Shape: {final_array.shape}")
            return final_array

        except Exception as e:
            logger.error(f"Erro no pipeline PIL: {e}")
            raise

    def get_image_info(self, processed_array):
        """
        Retorna informações sobre o array processado
        """
        return {
            "shape": processed_array.shape,
            "dtype": str(processed_array.dtype),
            "min_value": float(np.min(processed_array)),
            "max_value": float(np.max(processed_array)),
            "mean_value": float(np.mean(processed_array))
        }