# 1. Load QA Dataset (made-up by myself)
from datasets import Dataset
data = {
		"prompt": [
				"Can a tenant break a lease early in California?",
        "What is the statute of limitations for a contract breach in New York?",
        "..."
		],
		"response": [
				"Under California law, tenants may break a lease early in cases such as domestic violence or health issues.",
        "In New York, the statute of limitations for a contract breach is generally six years.",
        "..."
		]
}
dataset = Dataset.from_dict(data) # compatible with HuggingFace (more powerful features like batching, processing)

# 2. Load Pretrained Model and Tokenizer (tify-gpt2, Llama-2-7b-hf)
from transformers import AutoTokenizer
model_name = "sshleifer/tiny_gpt2" # small model for demo, "meta-llama/Llama-2-7b-hf" for full fine-tune
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # aviod padding issue

# 3. Preprocessing and Tokenization
# structured QA formatting to instruct model behavior (SFT,PEFT) (when see "question" generate "answer")
def format(example):
		full_prompt = f"Question: {example['prompt']} \n Answer: {example['response']}" # hard prompt
		tokens = tokenizer(full_prompt, truncation=True, padding="max_length", max_length=128) # no need return_tensor, cus map only accepts default, trainer converts to tensor implicitly
		tokens["label"] = tokens["input_ids"].copy() # for next token prediction (casual)
		return tokens
tokenized_dataset = dataset.map(format)

# observe for experiments
tokenizer.decode(tokenized_dataset[0]["input_ids"], skip_special_tokens=True) # restore actural input prompt text
tokenized_dataset[0] # observe first one in tokenized data
# {
#		'input_ids': [318, 25, 264, 1031, 718, ...] # length 128
#  	'attention_mask': [1, 1, 1,...,0 ,0 ], # 1 for real, 0 for padding
#	 	'labels': [...] # same as input_ids so model learns to predict the next token (autoregressive)
# }
# input_ids -> subword units, "lawful" -> "law", "ful"
# attention_mask and labels -> autoregressive (learn to predict next token)

# 4. Fine-Tune with LoRA
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCasualLM
lora_config = LoraConfig(
		task_type=TaskType.CASUAL_LM, # casual prediction (autoregressive) -> QA tasks
		r=4,
		lora_alpha=16, # scale
		lora_dropout=0.1,
		target_modules=["q_proj", "v_proj"], # injection target
		bias=None
)
base_model = AutoModelForCasualLM.from_pretrained(model_name)
model = get_peft_model(base_model, lora_config) # wrap up base model with

model.print_trainable_parameters() # observation

# 5. Training Parameters
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
		output_dir="./outputs",
		num_train_epochs=3,
		logging_steps=1,
		save_steps=10,
		save_total_limit=1,
		per_device_train_batch_size=2
)

trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_dataset,
		#data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()

# 6. Evaluation (automatic, human)
input_prompts = [prompt1, prompt2, ...]
references = [[ref1], [ref2], ...] # List[List[str]], bc of how evaluation functions are designed
predictions = [] # List[str]
model.eval()
for input_prompt in input_prompts: # "Question: What is the penalty for perjury in California? \n Answer:"
		input_tokens = tokenizer(input_prompt, return_tensors='pt'/'tf') # convert to tensors when serving (automatic batching, GPU acceleration)
		with torch.no_grad(): # alwasy use during inference, save memory (no track gradients)
				output_tokens = model.generate(input_tokens["input_ids"], max_new_tokens=128)
		output_response = tokenizer.decode(output_tokens[0], skip_special_tokens=True) # [0] to unpack tensor data
		# skip_special_tokens : <pad>, <s>/<bos> beginning, </s>/<eos> end of sentence, <unk>, <mask>
		predictions.append(output_response)

# EM
def exact_match_score(pred, ref):
		return int(pred.strip().lower() == ref.strip().lower())

em_scores = [exact_match_score(p, r[0]) for p, r in zip(predictions, references)]
print("Exact Match:", np.mean(em_scores)) # strict

# BLUE, ROGUE
import evaluate
bleu = evaluate.load("bleu")
bleu_result = bleu.compute(predictions=predictions, references=references)
print("BLEU:", bleu_result["bleu"])

rogue = evaluate.load("rogue")
rogue_result = rogue.compute(predictions=predictions, references=[r[0] for r in references]) # flatten ref List[str]
print("ROGUE-1", bleu_result["rogue1"])
print("ROGUE-L", bleu_result["rogueL"])

# 7. Deployment
# Save/Load fine-tuned model
model.save_pretrained('legal-qa-model')
tokenizer.save_pretrained('legal-qa-model')