from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="tiiuae/falcon-7b-instruct", filename="config.json")

app = Flask(__name__)

@app.route('/falcon', methods=['POST'])

def falcon():
    repo_id = "tiiuae/falcon-7b-instruct"
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64})
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    print(llm_chain.run(question))