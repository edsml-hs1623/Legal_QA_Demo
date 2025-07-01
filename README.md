# Legal_QA_Demo
Demonstration of a fine-tuned LLM model for legal QA using LoRA.
For demo purpose only and not executable yet due to resource constraints.

## Description
**QA Assistant**
- Task -> Autoregressive (Casual/Next-Token Prediction)
- Data -> Prompt-Response Pairs
- Model -> Fine-tuning with LoRA (SFT,PEFT)

## Steps
Tokenization Pipeline
1. Load QA Dataset (made-up dataset)
2. Load Pretrained Model and Tokenizer (tify-gpt2)
3. Preprocessing/Formatting and Tokenization
Fine-Tuning
4. Fine-Tune with LoRA (LoraConfig, get_peft_model)
5. Training Parameters (TrainingArguments, Trainer)
Evalulation and Deployment
6. Evaluation (EM, BLEU)
7. Deployment (FastAPI, Docker, k8s, local/cloud)

## Structure
legal_qa/
├── app/
│   └── main.py
├── legal_qa_model/
├── Dockerfile
├── requirements.txt
├── README.md
├── LICENSE
└── start.sh

## Getting Started 

1. make the shell script executable
```bash
chmod +x start.sh 
```
2. test locally (build and run container, send a test request)
```bash
docker build -t legal_qa .  
docker run -p 8000:8000 legal_qa

curl -X POST http://localhost:8000/qa -H "Content-Type: application/json" -d '{"question": "What is the penalty for perjury in California?"}'
```

3. cloud deployment (AWS EC2, ssh to instance, scp project)
```bash
ssh -i your-key.pem ec2-user@your-public-ip # SSH to EC2 instance
scp -i your-key.pem -r ./legal_qa ec2-user@your-ip:/home/ec2-user # security copy, clone

cd legal_qa
docker build -t legal-qa .
docker run -d -p 8000:8000 legal-qa
```

4. deployment with elatstic kubernetes service
```bash
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin # build and push docker image to GCR
<your-aws-account-id>.dkr.ecr.<region>.amazonaws.com
docker build -t legal-qa .
docker tag legal-qa:latest <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/legal-qa:latest
docker push <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/legal-qa:latest

eksctl create cluster --name legal-qa-cluster --region <region> --nodes 3 # create EKS clusters

kubectl apply -f eks-deployment.yaml # deploy to EKS
kubectl apply -f eks-service.yaml

kubectl get service legal-qa-service # get external ip
```
