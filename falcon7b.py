from flask import Flask, jsonify, request
from langchain import HuggingFacePipeline, PromptTemplate,  LLMChain
from transformers import AutoTokenizer, pipeline
import torch


app = Flask(__name__)

@app.route('/falcon', methods=['POST'])

def falcon():
    text = request.json['text']
    model_name = 'tiiuae/falcon-7b-instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    pipeline = pipeline(
        "text-generation", #task
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

    template = """
    You are an intelligent chatbot. Help the following question with brilliant answers.
    Question: {text}
    Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["text"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    generated_text = llm_chain.run(text)
    print(generated_text)
    response = jsonify({'generated_text': generated_text})
    return response

if __name__ == '__main__':
    app.run(debug=True)