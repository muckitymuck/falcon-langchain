from flask import Flask, jsonify, request
from langchain import HuggingFacePipeline, PromptTemplate,  LLMChain
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

@app.route('/falcon', methods=['POST'])

def falcon():
    text = request.json['text']
    model_name = 'TheBloke/falcon-7b-instruct-GPTQ'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model_basename = "gptq_model-4bit-64g"
    model = AutoGPTQForCausalLM.from_quantized(model_name,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

    prompt = text
    prompt_template=f'''A helpful assistant who helps the user with any questions asked.
    User: {prompt}
    Assistant:'''
    print("\n\n*** Generate:")

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
    print(tokenizer.decode(output[0]))
    logging.set_verbosity(logging.CRITICAL)

    print("*** Pipeline:")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    print(pipe(prompt_template)[0]['generated_text'])
    generated_text = pipe(prompt_template)[0]['generated_text']

    response = jsonify({'generated_text': generated_text})
    return response

if __name__ == '__main__':
    app.run(debug=True)