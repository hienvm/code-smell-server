import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from flask import Flask, jsonify, request
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pickle
import openai
import json
from xgboost import XGBClassifier


app = Flask("Code Smells Detection")

client_openai = openai.OpenAI()

with open('./prompts/analysis.json', 'r') as file:
    analysis_prompts = json.load(file)
with open('./prompts/refactor.json', 'r') as file:
    refactor_prompts = json.load(file)
# with open('./prompts/refactor.json', 'r') as file:
#     analyze_prompts = json.load(file)

models = {}


code_smells = [
    'data_class',
    'feature_envy',
    'complex_conditional',
    'complex_method'
    ]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoders = {}
classifiers = {}
thresholds = {}
tokenizer = AutoTokenizer.from_pretrained('microsoft/unixcoder-base', padding_side="right")
for smell in code_smells:
    encoders[smell] = AutoModelForSequenceClassification.from_pretrained(f'models/{smell}/unixcoder', use_safetensors=True).roberta
    encoders[smell].to(device)
    encoders[smell].eval()
    classifiers[smell] = XGBClassifier()
    classifiers[smell].load_model(f'models/{smell}/xgb.json')
    with open(f'models/{smell}/threshold.txt', 'r') as file:
        thresholds[smell] = float(file.read())

    
    
# detect_cache = {}
# for smell in code_smells:
#     detect_cache[smell] = {}
def avg_pool(token_emb, mask):
    return ((token_emb * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)).detach().cpu()

def tokenize(text):
    outputs = tokenizer(text, truncation=True, padding="max_length", max_length=1024)
    return outputs

def detect(smell, sourceCode):
    with torch.no_grad():
        tokens = tokenize(sourceCode)
        out = encoders[smell](input_ids=torch.tensor(tokens['input_ids']).to(device).reshape(1,-1), attention_mask=torch.tensor(tokens['attention_mask']).to(device).reshape(1,-1))
        emb: torch.Tensor = avg_pool(out.last_hidden_state, torch.tensor(tokens['attention_mask']).to(device).reshape(1,-1) )
        proba = classifiers[smell].predict_proba(emb.detach().numpy())[:,1].item()
    return int(proba > thresholds[smell])


@app.route("/detect", methods=["POST"])
def detectCodeSmells():
    response = {}
    try:
        body = request.get_json(force=True)
        for smell in code_smells:
            if body['type'] == smell:
                response['label'] = detect(smell, body['sourceCode'])
                print(smell + ' ' + str(response['label']))
                break
        
    except Exception as e:
        print(e)
        print("Something's wrong")
        return jsonify({"msg": "Error"})
    
    return jsonify(response)

@app.route("/analyze", methods=['POST'])
def analyzeCodeSmells():
    response = {}
    try:
        body = request.get_json(force=True)
        for smell in code_smells:
            if body['type'] == smell:
                print('Analyzing ' + smell)
                analysis = client_openai.responses.create(
                    model='gpt-4.1-mini',
                    instructions=analysis_prompts[smell]['instructions']+'\n'+body['sourceCode'],
                    input=analysis_prompts[smell]['input'],
                    max_output_tokens=3000
                )
                response['analysis'] = analysis.output_text
                break
        
    except Exception as e:
        print(e)
        print("Something's wrong")
        return jsonify({"msg": "Error"})
    
    return jsonify(response)

@app.route("/refactor", methods=['POST'])
def refactorCodeSmells():
    response = {}
    try:
        body = request.get_json(force=True)
        for smell in ['complex_conditional', 'complex_method']:
            if body['type'] == smell:
                print('Refactoring ' + smell)
                analysis = client_openai.responses.create(
                    model='gpt-4.1',
                    instructions=refactor_prompts[smell]['instructions']+'\n'+body['sourceCode'],
                    input=refactor_prompts[smell]['input'],
                )
                response['refactoredCode'] = analysis.output_text
                print(response)
                # response['refactoredCode'] = '''
                # ```java
                # public void fun1(){return;}
                # ```
                # ```java
                # public void fun2(){return;}
                # ```
                # '''
                break
        
    except Exception as e:
        print(e)
        print("Something's wrong")
        return jsonify({"msg": "Error"})
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)