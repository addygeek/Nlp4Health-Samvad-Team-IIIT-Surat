#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-10-14T06:12:38.390Z
"""

# ==============================================================================
# Evaluation Script for the Fine-Tuned mT5 Summarization Model
# ==============================================================================
# This script loads your final, trained model, generates summaries for the 'dev'
# set, and calculates the official ROUGE, BLEU, and BERTScore metrics to
# measure the model's performance.
# ==============================================================================

# --- Step A: Install necessary libraries ---
!pip install --upgrade transformers datasets evaluate rouge_score bert_score sentencepiece -q

# --- Step B: Import all required packages ---
import os
import glob
import json
import torch
import evaluate
from google.colab import drive
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

# --- Step C: Mount your Google Drive ---
print("Mounting Google Drive...")
drive.mount('/content/drive')

# --- Step D: Load your trained model and tokenizer ---
print("\nLoading your fine-tuned model...")
# Path to the final checkpoint folder of your trained model
model_path = "/content/drive/My Drive/NLP_A14_Health/mT5_Baseline_Model/checkpoint-9135"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True).to(device)
print("✅ Model loaded successfully!")


# --- Step E: Prepare the 'dev' set for evaluation ---
print("\nPreparing the 'dev' set for evaluation...")
dataset_path = "/content/dataset"
zip_file_path = "/content/drive/My Drive/NLP_A14_Health/SharedTask_NLPAI4Health_Train&dev_set.zip"
if not os.path.exists(dataset_path):
    print(f"\nDataset folder not found. Unzipping from {zip_file_path}...")
    !unzip -q "{zip_file_path}" -d "/content/dataset/"
    print("✅ Unzip complete.")
else:
    print("\nDataset folder already exists.")

base_path = "/content/dataset/SharedTask_NLPAI4Health_Train&dev_set/"
# IMPORTANT: We are now using the 'dev' folder for evaluation
dev_dialogue_files = glob.glob(os.path.join(base_path, 'dev', '*', 'Dialogues', '*.jsonl'))
dev_summary_files = glob.glob(os.path.join(base_path, 'dev', '*', 'Summary_Text', '*_summary.txt'))

grouped_dev_files = {}
for file_path in dev_dialogue_files:
    filename = os.path.basename(file_path)
    unique_id = filename.replace(".jsonl", "")
    if unique_id not in grouped_dev_files: grouped_dev_files[unique_id] = {}
    grouped_dev_files[unique_id]['dialogue'] = file_path

for file_path in dev_summary_files:
    filename = os.path.basename(file_path)
    unique_id = filename.replace("_summary.txt", "")
    if unique_id in grouped_dev_files: grouped_dev_files[unique_id]['summary'] = file_path

dev_processed_data = []
for unique_id, files in grouped_dev_files.items():
    if 'dialogue' in files and 'summary' in files:
        try:
            dialogue_turns = []
            with open(files['dialogue'], 'r', encoding='utf-8') as f:
                for line in f:
                    turn = json.loads(line)
                    dialogue_turns.append(f"{turn.get('speaker', 'N/A')}: {turn.get('dialogue', '')}")
            full_dialogue_text = "\n".join(dialogue_turns)

            summary_text = open(files['summary'], 'r', encoding='utf-8').read()

            if full_dialogue_text and summary_text:
                dev_processed_data.append({'dialogue': full_dialogue_text, 'summary': summary_text})
        except:
            continue

dev_dataset = Dataset.from_list(dev_processed_data)
# We will test on a sample of 100 examples to get a quick result.
# For the final paper, you can remove the next line to run on the full dev set.
evaluation_sample = dev_dataset.select(range(100))
print(f"✅ Created a sample dev set with {len(evaluation_sample)} examples.")


# --- Step F: Generate summaries for the evaluation sample ---
print("\nGenerating predictions from the model...")
prefix = "summarize: "
predictions = []
references = []

for item in tqdm(evaluation_sample, desc="Generating Summaries"):
    input_text = prefix + item["dialogue"]
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)

    output_ids = model.generate(
        inputs["input_ids"],
        max_length=256,
        num_beams=4,
        early_stopping=True
    )
    generated_summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    predictions.append(generated_summary)
    references.append(item["summary"])


# --- Step G: Calculate and Print the Scores ---
print("\nCalculating scores...")
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

rouge_results = rouge.compute(predictions=predictions, references=references)
bleu_results = bleu.compute(predictions=predictions, references=references)
bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="en") # Using 'en' as a default for multilingual models

print("\n" + "="*50)
print("BASELINE MODEL EVALUATION RESULTS:")
print("="*50)
print(f"ROUGE-1: {rouge_results['rouge1'] * 100:.2f}")
print(f"ROUGE-2: {rouge_results['rouge2'] * 100:.2f}")
print(f"ROUGE-L: {rouge_results['rougeL'] * 100:.2f}")
print("-" * 20)
print(f"BLEU Score: {bleu_results['bleu'] * 100:.2f}")
print("-" * 20)
# BERTScore prints a list, so we calculate the average F1 score
avg_bertscore_f1 = sum(bertscore_results['f1']) / len(bertscore_results['f1'])
print(f"BERTScore (F1): {avg_bertscore_f1 * 100:.2f}")
print("="*50)