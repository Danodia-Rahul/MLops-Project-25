from flask import Flask, request, jsonify
import pandas as pd
from predict import predict

app = Flask(__name__)


@app.route('/predict', methods=['Post'])
def predict_route():
    try:
        data = request.get_json()
        inputs = pd.DataFrame(data)

        preds = predict(inputs)
        return jsonify({'prediction': preds})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)