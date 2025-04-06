import numpy as np
import joblib

class Model:
    # filepath: /home/khushijha/Workspace/WaterAnalysisSupervised/src/Model.py
    def __init__(self):

        # Load the trained SVM model
        self.model_path = "./models/svm_model.joblib"
        self.svm_model = joblib.load(self.model_path)

        # Load the scaler and label encoder used during training
        self.scaler_path = "./models/scaler.joblib"
        self.label_encoder_path = "./models/label_encoder.joblib"
        self.scaler = joblib.load(self.scaler_path)
        self.label_encoder = joblib.load(self.label_encoder_path)

    # Function to predict use case
    def predict(self, parameters):
        input_data = np.array(parameters).reshape(1, -1)
        input_scaled = self.scaler.transform(input_data)
        prediction = self.svm_model.predict(input_scaled)
        result = self.label_encoder.inverse_transform(prediction)[0]
        print("Prediction Result: ", result)

        return result

# Example usage
# example_parameters = [7.38, 2.2456, -10.10, 1.15, 8572.40, 4380.50, 11510.50, 2155.30, 98.50, 1025.30, 612.30, 11820.40, 2185.30]
# predicted_use_case = predict_usecase(example_parameters)
# print("Predicted Use Case:", predicted_use_case)