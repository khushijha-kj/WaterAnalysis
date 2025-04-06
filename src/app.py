from flask import Flask, jsonify, request
from Model import Model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"status":"ok", "message": "hello world"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.get_json()
    try:
        if not input_data:
            return jsonify({"error": "Invalid input data"}), 400
        
        # Extract data from JSON and format it as a list
        parameters = [
            input_data.get("PH"),
            input_data.get("EC"),
            input_data.get("ORP"),
            input_data.get("DO"),
            input_data.get("TDS"),
            input_data.get("TSS"),
            input_data.get("TS"),
            input_data.get("TOTAL_N"),
            input_data.get("NH4_N"),
            input_data.get("TOTAL_P"),
            input_data.get("PO4_P"),
            input_data.get("COD"),
            input_data.get("BOD"),
        ]

        # Ensure all parameters are present
        if None in parameters:
            return jsonify({"error": "Missing parameters in input data"}), 400
        
        model = Model()
        result = model.predict(parameters)

        return jsonify({"prediction": result}), 200
        
    except Exception as e:
        return jsonify({"error": "Error processing input data","e" : e}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)