from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCasualLM
import torch

app = FastAPI()

model = AutoModelForCasualLM.from_pretrained('legal-qa-model')
tokenizer = AutoTokenizer.from_pretrained('legal-qa-model')
model.eval()

@app.post('/qa')
async def answer_question(req: Request)：
		body = await req.json()
		question = body["question"]
		
		input_prompt = f"Question: {question}\n Answer:"
		input_tokens = tokenizer(input_prompt, return_tensors='pt')
		with torch.no_grad():
				output_tokens = model.generate(input_tokens["input_ids"], skip_special_tokens=True)
		output_response = tokenizer.decode(output_tokens[0])
		return f"Answer: {output_response}"
