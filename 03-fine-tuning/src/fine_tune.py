# --- Imports --- #

import os
import sys

# Add root dir to the first index in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import logging

import numpy as np

from config import (
    ENCODER_BASE_MODEL,
    EMBEDDING_MODEL_PATH,
    TRAINING_DATA_PATH,
)

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)

from datasets import Dataset
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import TripletDistanceMetric, TripletLoss

from torch import cuda

# --- Logger --- #
logging.basicConfig(level = logging.INFO, format="%(message)s")

logger = logging.getLogger(__name__)

# --- Load Training Data --- #

def load_training_data(path):

    logger.info(f"Loading Training Data...")
    print("-" * 50)
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Total Records Loaded: {len(data)}")
    print("-" * 50)

    return data


def create_datasets(data):
    
    np.random.seed(42)

    def gen():
        for d in data:
            yield d

    ds = Dataset.from_generator(gen)

    idx = np.random.permutation(len(data))

    split = int(len(idx) * 0.8)

    train_ix = idx[:split]
    valid_ix = idx[split:]

    train_ds = Dataset.from_dict(ds[train_ix])

    eval_ds = Dataset.from_dict(ds[valid_ix])

    return train_ds, eval_ds

def create_evaluator(validation_dataset):

    evaluator = TripletEvaluator(
        anchors=validation_dataset[:]['anchor'],
        positives=validation_dataset[:]['positive'],
        negatives=validation_dataset[:]['negative'],
        name='eva-triplet',
        write_csv=True
    )

    return evaluator

def fine_tune():
    device = "cuda" if cuda.is_available() else "cpu"

    logger.info(f"Device: {device}")
    print("-" * 50)

    data = load_training_data(TRAINING_DATA_PATH)

    MODEL = SentenceTransformer(ENCODER_BASE_MODEL, device=device)

    logger.info(f"Embedding Model: {MODEL}")
    print("-" * 50)
   
    logger.info(f"Creating Datasets...")
    print("-" * 50)
    
    TRAIN_DS, EVAL_DS = create_datasets(data)

    evaluator = create_evaluator(EVAL_DS)
    
    logger.info(f"Evaluating Base Model...")
    print("-" * 50)
    
    BASE_MODEL_RESULTS = evaluator(
        model = MODEL
    )

    BASE_MODEL_SCORE = BASE_MODEL_RESULTS[evaluator.primary_metric]

    print(f"Score Base Model: {BASE_MODEL_SCORE}")
    print("-" * 50)

    LOSS = TripletLoss(
        model = MODEL,
        distance_metric= TripletDistanceMetric.COSINE,
        triplet_margin = 0.2
    )

    MODEL_PATH = os.path.join(EMBEDDING_MODEL_PATH, "all-MiniLM-L6-v2-triplet/")
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    METRIC_KEY = "eval_eva-triplet_cosine_accuracy"

    TRAINING_ARGS = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=MODEL_PATH,
        # Optional training parameters:
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        bf16=cuda.is_available(),
        dataloader_pin_memory = cuda.is_available(),
        # Optional tracking/debugging parameters:
        logging_steps = 5,
        save_strategy = "epoch",
        eval_strategy = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = METRIC_KEY
    )

    TRAINER = SentenceTransformerTrainer(
        model = MODEL,
        args = TRAINING_ARGS,
        train_dataset = TRAIN_DS,
        eval_dataset = EVAL_DS,
        loss = LOSS,
        evaluator = evaluator
    )

    logger.info(f"Training Initialized...")
    print("-" * 50)
    
    TRAINER.train()

    logger.info(f"Model Evaluation after Fine-Tuning...")
    print("-" * 50)

    FT_MODEL_RESULTS = evaluator(
        model = MODEL
    )

    FT_MODEL_SCORE = FT_MODEL_RESULTS[evaluator.primary_metric]
    
    print("-" * 50)
    print(f"Fine-Tuned Model Score: {FT_MODEL_SCORE}")
    print("-" * 50)
    
    DELTA = FT_MODEL_SCORE - BASE_MODEL_SCORE

    print(f"Δ: {DELTA:.4f}")
    print("-" * 50)

    BEST_MODEL_PATH = os.path.join(EMBEDDING_MODEL_PATH, "best-model/")
    os.makedirs(BEST_MODEL_PATH, exist_ok=True)
    
    MODEL.save(BEST_MODEL_PATH)
    print(f"Model Saved")
    print("-" * 50)
    print(f"End of script")

    return MODEL

if __name__ == "__main__":
    fine_tune()










    




