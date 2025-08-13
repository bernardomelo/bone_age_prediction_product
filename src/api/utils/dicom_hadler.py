import pydicom
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

# LABORATÓRIO PROVAVELMENTE NÃO UTILIZARÁ DICOM
class DicomHandler:
    """
    Classe para manipulação de arquivos DICOM
    """

    def __init__(self):
        self.supported_extensions = ['.dcm', '.dicom']

    def is_dicom_file(self, file_path_or_bytes):
        """
        Verifica se o arquivo é um DICOM válido
        """
        try:
            if isinstance(file_path_or_bytes, (str, bytes)):
                pydicom.dcmread(file_path_or_bytes)
                return True
            return False
        except Exception as e:
            logger.warning(f"Arquivo não é DICOM válido: {e}")
            return False

    def read_dicom(self, file_path_or_bytes):
        """
        Lê arquivo DICOM e retorna o dataset
        """
        try:
            dicom_data = pydicom.dcmread(file_path_or_bytes)
            logger.info(f"DICOM lido com sucesso")
            return dicom_data
        except Exception as e:
            logger.error(f"Erro ao ler DICOM: {e}")
            raise ValueError(f"Não foi possível ler o arquivo DICOM: {e}")

    def extract_image_array(self, dicom_data):
        """
        Extrai array de pixels da imagem DICOM
        """
        try:
            if not hasattr(dicom_data, 'pixel_array'):
                raise ValueError("DICOM não contém dados de imagem")

            # Obter array de pixels
            pixel_array = dicom_data.pixel_array

            # Aplicar transformações se necessário
            if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
                slope = dicom_data.RescaleSlope
                intercept = dicom_data.RescaleIntercept
                pixel_array = pixel_array * slope + intercept

            logger.info(f"Array extraído - Shape: {pixel_array.shape}")
            return pixel_array

        except Exception as e:
            logger.error(f"Erro ao extrair imagem: {e}")
            raise ValueError(f"Não foi possível extrair imagem do DICOM: {e}")

    def normalize_image(self, pixel_array):
        """
        Normaliza o array de pixels para 0-255
        """
        try:
            # Converter para float para evitar overflow
            pixel_array = pixel_array.astype(np.float64)

            # Normalizar para 0-255
            min_val = np.min(pixel_array)
            max_val = np.max(pixel_array)

            if max_val > min_val:
                normalized = ((pixel_array - min_val) / (max_val - min_val) * 255)
            else:
                normalized = pixel_array

            # Converter para uint8
            normalized = normalized.astype(np.uint8)

            logger.info("Imagem normalizada")
            return normalized

        except Exception as e:
            logger.error(f"Erro na normalização: {e}")
            raise ValueError(f"Erro ao normalizar imagem: {e}")

    def array_to_pil(self, pixel_array):
        """
        Converte array numpy para PIL Image
        """
        try:
            # Se for 3D, pegar a primeira fatia
            if len(pixel_array.shape) == 3:
                pixel_array = pixel_array[0]

            # Criar imagem PIL
            pil_image = Image.fromarray(pixel_array)

            # Converter para RGB se necessário
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            logger.info(f"Imagem PIL criada - Modo: {pil_image.mode}, Tamanho: {pil_image.size}")
            return pil_image

        except Exception as e:
            logger.error(f"Erro ao converter para PIL: {e}")
            raise ValueError(f"Erro ao converter array para PIL: {e}")

    def process_dicom_to_image(self, file_path_or_bytes):
        """
        Pipeline completo: DICOM -> PIL Image
        """
        try:
            # Ler DICOM
            dicom_data = self.read_dicom(file_path_or_bytes)

            # Extrair array de pixels
            pixel_array = self.extract_image_array(dicom_data)

            # Normalizar
            normalized_array = self.normalize_image(pixel_array)

            # Converter para PIL
            pil_image = self.array_to_pil(normalized_array)

            # Metadados úteis
            metadata = self.extract_metadata(dicom_data)

            logger.info("Processamento DICOM completo")
            return pil_image, metadata

        except Exception as e:
            logger.error(f"Erro no processamento completo: {e}")
            raise

    def extract_metadata(self, dicom_data):
        """
        Extrai metadados relevantes do DICOM
        """
        metadata = {}

        try:
            # Informações do paciente (se disponíveis)
            metadata['patient_age'] = getattr(dicom_data, 'PatientAge', 'N/A')
            metadata['patient_sex'] = getattr(dicom_data, 'PatientSex', 'N/A')

            # Informações da imagem
            metadata['rows'] = getattr(dicom_data, 'Rows', 0)
            metadata['columns'] = getattr(dicom_data, 'Columns', 0)
            metadata['pixel_spacing'] = getattr(dicom_data, 'PixelSpacing', [0, 0])

            # Informações do equipamento
            metadata['manufacturer'] = getattr(dicom_data, 'Manufacturer', 'N/A')
            metadata['model_name'] = getattr(dicom_data, 'ManufacturerModelName', 'N/A')

            # Data do estudo
            metadata['study_date'] = getattr(dicom_data, 'StudyDate', 'N/A')

        except Exception as e:
            logger.warning(f"Erro ao extrair metadados: {e}")

        return metadata

    def save_image_as_jpg(self, pil_image, output_path):
        """
        Salva imagem PIL como JPG
        """
        try:
            # Garantir modo RGB
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Salvar como JPG
            pil_image.save(output_path, 'JPEG', quality=95)
            logger.info(f"Imagem salva em: {output_path}")

        except Exception as e:
            logger.error(f"Erro ao salvar imagem: {e}")
            raise ValueError(f"Erro ao salvar imagem: {e}")

    def image_to_bytes(self, pil_image, format='JPEG'):
        """
        Converte PIL Image para bytes
        """
        try:
            img_buffer = io.BytesIO()

            # Garantir modo RGB para JPEG
            if format.upper() == 'JPEG' and pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            pil_image.save(img_buffer, format=format, quality=95)
            img_bytes = img_buffer.getvalue()

            logger.info(f"Imagem convertida para bytes - Tamanho: {len(img_bytes)}")
            return img_bytes

        except Exception as e:
            logger.error(f"Erro ao converter para bytes: {e}")
            raise ValueError(f"Erro ao converter imagem para bytes: {e}")