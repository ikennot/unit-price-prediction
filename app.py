from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

MODEL_PATH = 'model/new_unit_price_model.joblib'
SCALER_PATH = 'model/new_scaler.joblib'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/predict', methods=['POST'])
def predict_unit_price():
    try:
        data = request.get_json()

        bathroom = float(data.get('bathroom'))
        bedroom = float(data.get('bedroom'))
        floor_area = float(data.get('floor_area'))
        lot_size = float(data.get('lot_size'))
        start_year = int(data.get('year'))
        n_years = int(data.get('n_years', 5))

        predictions = []
        base_price = None

        for i in range(n_years):
            y = start_year + i
            features = [[bathroom, bedroom, floor_area, lot_size, y]]
            scaled = scaler.transform(features)
            price = model.predict(scaled)[0]

            # optional: apply 6% growth per year
            if base_price is None:
                base_price = price
            else:
                price = base_price * ((1.06) ** i)

            predictions.append({
                'year': y,
                'predicted_price': round(price, 2)
            })

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False)
