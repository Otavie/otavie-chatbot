
from flask import Flask, render_template, request, jsonify
from chat import get_response

app = Flask(__name__)

# @app.get("/")
@app.route("/")
def index():
    return render_template("index.html")

# @app.post("/predict")
@app.route("/predict", methods=["POST"])
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    print(response)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)