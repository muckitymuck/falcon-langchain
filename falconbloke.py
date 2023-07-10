import json
from flask import Flask, request, stream_with_context
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

app = Flask(__name__)

@app.route('/falcon', methods=['POST'])
def generate():
    model_name_or_path = "TheBloke/falcon-7b-instruct-GPTQ"
    model_basename = "gptq_model-4bit-64g"
    use_triton = False

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=use_triton,
            quantize_config=None)

    @stream_with_context
    def generate_response():
        request_data = request.get_json()
        prompt = request_data.get('prompt', '')
        prompt_template=f'''A helpful assistant who helps the user with any questions asked.
        User: {prompt}
        Assistant:'''

        input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
        output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
        generated_text = tokenizer.decode(output[0])

        # Inference can also be done using transformers' pipeline
        # Note that if you use pipeline, you will see a spurious error message saying the model type is not supported
        # This can be ignored!  Or you can hide it with the following logging line:
        # Prevent printing spurious transformers error when using pipeline with AutoGPTQ
        logging.set_verbosity(logging.CRITICAL)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        generated_text_pipeline = pipe(prompt_template)[0]['generated_text']

        response = {
            'generated_text': generated_text,
            'generated_text_pipeline': generated_text_pipeline
        }

        yield json.dumps(response).encode()

    return generate_response()

if __name__ == '__main__':
    app.run(debug=True)