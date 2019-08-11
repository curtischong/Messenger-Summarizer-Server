import json
from flask import Flask, request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/get_phrases', methods=['GET'])
def get_phrases():
  print(request.args.get("messages"))
  return 'Hello, World!'

if __name__ == "__main__":
  app.run()