# Creating webservice for our model with Flask #
from flask import Flask, request, jsonify
import pickle

app = Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
    return "PONG"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696) # localhost:9696/ping

# in terminal
# ipython
# import ping -> ping.ping()