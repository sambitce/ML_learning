#!flask/bin/python
from flask import Flask,jsonify,request
import call_model


app = Flask(__name__)

@app.route('/forecast' , methods=['GET'])
def index():
    date1 = request.args.get('forecast_date')
    return jsonify({'forecast': str(call_model.forecast(date1))})
    
if __name__ == '__main__' :
    app.run()
