from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

@app.route('/falcon', methods=['POST'])
def falcon():
    text = request.json['text']
    model_name = '/home/ubuntu/falcon-7b-instruct//'  # Replace with the path to your LLM model directory

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    input_ids = tokenizer.encode(text, return_tensors="pt")
    generated_ids = model.generate(input_ids, max_length=200, num_return_sequences=1)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    response = jsonify({'generated_text': generated_text})
    return response

if __name__ == '__main__':
    app.run(debug=True)
