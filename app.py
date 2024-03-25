from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
import os
import prediction

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

class GetPredictionOutput(Resource):
    def post(self):
        try:
            data = request.get_json()
            title = data.get("title")
            stars = data.get("stars")
            reviews = data.get("reviews")
            isBestSeller = data.get("isBestSeller")
            boughtInLastMonth = data.get("boughtInLastMonth")
            predict = prediction.predict_price(title, stars, reviews, isBestSeller, boughtInLastMonth)
            predictOutput = predict.tolist() 
            return {'predict':predictOutput}

        except Exception as error:
            print("Error:", error)
            return {'error': 'An unexpected error occurred.'}

api.add_resource(GetPredictionOutput, '/getPredictionOutput')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
