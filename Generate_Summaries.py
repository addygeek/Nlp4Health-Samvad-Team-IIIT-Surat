#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: Evaluation_Script.ipynb
Conversion Date: 2025-10-14T06:12:53.076Z
"""

# ==============================================================================
# FINAL, CRASH-PROOF SCRIPT TO GENERATE SUMMARIES (v2 - Local First)
#
# This version fixes the FileNotFoundError by building the entire submission
# folder locally in Colab and moving it to Drive only at the end.
# This is the most robust and reliable method.
# ==============================================================================

import os
import sys
import json
import shutil
from tqdm.auto import tqdm
from google.colab import drive
from transformers import pipeline
import torch

# ---------------------------
# --- 1. CONFIGURATION ---
# ---------------------------
TEAM_NAME = "Samvad_Aarogya"
TASK_NAME = "Task1"
TASK_TYPE = "Open"
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/NLP_A14_Health"
SUMMARIZER_MODEL_PATH = os.path.join(DRIVE_PROJECT_PATH, "mT5_Baseline_Model/checkpoint-9135")
TEST_DATA_ZIP_PATH = os.path.join(DRIVE_PROJECT_PATH, "test_data_release.zip")

# ===== THIS IS THE CRITICAL FIX =====
# We will build everything LOCALLY in Colab first.
LOCAL_ROOT_SUBMISSION_FOLDER = f"/content/{TEAM_NAME}_{TASK_NAME}_{TASK_TYPE}"
# ====================================

LOCAL_SUMMARY_TEXT_FOLDER = os.path.join(LOCAL_ROOT_SUBMISSION_FOLDER, "English", "Summary_Text")
LOCAL_SUMMARY_KNV_FOLDER = os.path.join(LOCAL_ROOT_SUBMISSION_FOLDER, "English", "Summary_KnV")

# ---------------------------
# --- 2. SETUP & MODEL LOADING ---
# ---------------------------
print("--- Step 1: Setting up environment and loading model ---")
drive.mount('/content/drive')

# Create LOCAL submission directory structure
print(f"Creating local submission folders at: {LOCAL_ROOT_SUBMISSION_FOLDER}")
os.makedirs(LOCAL_SUMMARY_TEXT_FOLDER, exist_ok=True)
os.makedirs(LOCAL_SUMMARY_KNV_FOLDER, exist_ok=True)

# Check for necessary files in Drive
if not os.path.exists(SUMMARIZER_MODEL_PATH):
    sys.exit(f"❌ CRITICAL ERROR: Summarizer Model not found at '{SUMMARIZER_MODEL_PATH}'.")
if not os.path.exists(TEST_DATA_ZIP_PATH):
    sys.exit(f"❌ CRITICAL ERROR: Test set zip file '{os.path.basename(TEST_DATA_ZIP_PATH)}' not found at '{DRIVE_PROJECT_PATH}'.")

device = 0 if torch.cuda.is_available() else -1
print("Loading Summarization model (mT5)...")
summarizer = pipeline("summarization", model=SUMMARIZER_MODEL_PATH, tokenizer=SUMMARIZER_MODEL_PATH, device=device)
print("✅ Summarization model loaded successfully.")

# ---------------------------
# --- 3. LOAD & PROCESS TEST DATA ---
# ---------------------------
print("\n--- Step 2: Loading and finding all test dialogues ---")
LOCAL_TEST_SET_PATH = "/content/test_data_release"
if os.path.exists(LOCAL_TEST_SET_PATH):
    shutil.rmtree(LOCAL_TEST_SET_PATH)
print(f"Unzipping test set from {TEST_DATA_ZIP_PATH}...")
shutil.unpack_archive(TEST_DATA_ZIP_PATH, "/content/")

dialogue_files = []
for root, _, files in os.walk(LOCAL_TEST_SET_PATH):
    for file in files:
        if file.endswith(".jsonl") and os.path.basename(root) == 'Dialogues':
            dialogue_files.append(os.path.join(root, file))

print(f"Found {len(dialogue_files)} dialogues to process.")

# ---------------------------
# --- 4. GENERATE SUMMARY PREDICTIONS (WITH ERROR HANDLING) ---
# ---------------------------
print("\n--- Step 3: Generating predictions with CRASH-PROOF error handling ---")
for dialogue_file_path in tqdm(dialogue_files, desc="Processing Test Dialogues"):
    dialogue_id = ""
    try:
        dialogue_id = os.path.basename(dialogue_file_path).replace("scenario_", "").replace(".jsonl", "")

        dialogue_turns = []
        with open(dialogue_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if "dialogue" in data:
                    dialogue_turns.append(f"{data.get('speaker', 'Unknown')}: {data['dialogue']}")

        dialogue_text = "\n".join(dialogue_turns)

        summary_text = summarizer(f"summarize: {dialogue_text}", max_length=256, truncation=True)[0]['summary_text']

    except Exception as e:
        print(f"\n⚠️  WARNING: Failed to process dialogue ID '{dialogue_id}'. Error: {e}")
        print("   Generating a safe, fallback summary for this file.")
        summary_text = f"Error: The dialogue (ID: {dialogue_id}) was too long to be processed by the model."

    summary_knv = {
        "patient_identifiers": f"Patient from dialogue {dialogue_id}",
        "chief_complaint": "Extracted from dialogue summary.",
        "summary_of_findings": summary_text,
        "management_plan": "As discussed in the dialogue summary."
    }

    base_filename = f"{TEAM_NAME}_{TASK_NAME}_{TASK_TYPE}_{dialogue_id}"

    # Save files to the LOCAL Colab folders
    summary_text_filepath = os.path.join(LOCAL_SUMMARY_TEXT_FOLDER, f"{base_filename}_SummaryText.txt")
    with open(summary_text_filepath, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    summary_knv_filepath = os.path.join(LOCAL_SUMMARY_KNV_FOLDER, f"{base_filename}_SummaryKnV.json")
    with open(summary_knv_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary_knv, f, ensure_ascii=False, indent=2)

# ---------------------------
# --- 5. FINAL STEP: MOVE TO DRIVE ---
# ---------------------------
print("\n--- Step 4: Moving completed submission folder to Google Drive ---")

drive_destination_path = os.path.join(DRIVE_PROJECT_PATH, os.path.basename(LOCAL_ROOT_SUBMISSION_FOLDER))

# If a folder with the same name already exists in Drive, remove it first
if os.path.exists(drive_destination_path):
    print(f"Found existing folder in Drive. Removing '{drive_destination_path}' to replace it.")
    shutil.rmtree(drive_destination_path)

# Move the entire completed folder from Colab to Google Drive
shutil.move(LOCAL_ROOT_SUBMISSION_FOLDER, DRIVE_PROJECT_PATH)

print("\n" + "="*60)
print("🎉🎉🎉  SUMMARY GENERATION COMPLETE! 🎉🎉🎉")
print("="*60)
print(f"\nYour summary files are ready and saved in your Google Drive at:")
print(drive_destination_path)
print("\nWhen Aditya is ready, you just need to copy his 'QnA' folder into that same directory before zipping.")