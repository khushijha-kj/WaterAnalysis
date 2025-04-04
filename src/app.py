from flask import Flask, jsonify, request
from Model import Model

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status":"ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.get_json()
    try:
        if not input_data:
            return jsonify({"error": "Invalid input data"}), 400
        
        # Extract data from JSON and format it as a list
        parameters = [
            input_data.get("param1"),
            input_data.get("param2"),
            input_data.get("param3"),
            input_data.get("param4"),
            input_data.get("param5"),
            input_data.get("param6"),
            input_data.get("param7"),
            input_data.get("param8"),
            input_data.get("param9"),
            input_data.get("param10"),
            input_data.get("param11"),
            input_data.get("param12"),
            input_data.get("param13"),
        ]

        # Ensure all parameters are present
        if None in parameters:
            return jsonify({"error": "Missing parameters in input data"}), 400
        
        model = Model()
        result = model.predict(parameters)

        return jsonify({"prediction": result}), 200
        
    except Exception as e:
        return jsonify({"error": "Error processing input data"}), 400

if __name__ == "__main__":
    app.run(debug=True)