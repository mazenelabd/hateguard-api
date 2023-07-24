from flask import Flask, request, jsonify, abort, make_response
from flask_cors import CORS
import numpy as np
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import os
import json
import torch
from dotenv import load_dotenv
import time

app = Flask(__name__)
CORS(app)

load_dotenv()
secret_key = os.getenv("SECRET_KEY")
model_name = os.getenv("model_name")
onnx_model = os.getenv("onnx_model_name")

id2label = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    if "Authorization" not in request.headers or request.headers["Authorization"] != secret_key:
        print(request.headers["Authorization"] != secret_key)
        print(request.headers["Authorization"])
        print(secret_key)
        abort(401)

    text_to_classify = request.json["data"]
    print("input:", text_to_classify)

    if isinstance(text_to_classify, str):
        text_to_classify = [text_to_classify]
    elif isinstance(text_to_classify, list) and all(isinstance(x, str) for x in text_to_classify):
        pass
    else:
        error_msg = "Input data must be a string or a list of strings."
        return jsonify(error=error_msg), 400
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    session = InferenceSession(onnx_model)
    inputs = tokenizer(text_to_classify, return_tensors="np", max_length=64, padding="max_length", truncation=True)
    inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
    logits = session.run(output_names=["logits"], input_feed=inputs)[0]

    outputs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()

    response = []
    for inner_list in outputs:
        label = id2label[max(range(len(inner_list)), key=inner_list.__getitem__)]
        response.append({id2label[i]: float(inner_list[i]) for i in range(len(inner_list))})
        response[-1]["label"] = label

    print("Response:", response)

    json_response = json.dumps(response)
    final_response = make_response(json_response)
    final_response.mimetype = "application/json"

    end_time = time.time()
    response_time = end_time - start_time
    print(f"Response time: {response_time:.3f} seconds")

    return final_response

@app.route("/")
def home():
    return "Welcome to the HateGuard API By Mazen Elabd"
