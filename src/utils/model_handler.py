from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import LocallyConnected2D
# from tensorflow.keras.metrics import mean_absolute_error
import numpy as np

# def _mae_months(in_gt, in_pred):
#     boneage_div = 1  # Use same value as in training
#     return mean_absolute_error(boneage_div * in_gt, boneage_div * in_pred)

class BoneAgeModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Loads model from model_path with the necessary attributes
        """
        self.model = load_model(self.model_path, custom_objects={
        # 'LocallyConnected2D': LocallyConnected2D,
        # 'mae_months': _mae_months
        },
        compile=False)
        print(f"- Model loaded from {self.model_path}")

    def predict(self, img_array) -> dict:
        """
        Real prediction with the model.
        :argument img_array: PIL Image object.
        """
        predicted_zscore = self.model.predict(img_array)

        boneage_mean = 0
        boneage_div = 1.0
        predicted_age = predicted_zscore * boneage_div + boneage_mean

        return {
            "predicted_age_months": round(predicted_age * 12, 1),
            "predicted_age_years": round(predicted_age, 1),
        }
