from flask import Flask, render_template, request, jsonify
from backend.models.tweeter_hashtag.use import predict_single_hashtag
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/hashtag')
def hashtag():
    return render_template('hashtag.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    hashtag = data.get('hashtag')

    if not hashtag:
        return jsonify({'error': 'Hashtag is required'}), 400

    try:
        probability = predict_single_hashtag(hashtag)
        return jsonify({'probability': probability})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)