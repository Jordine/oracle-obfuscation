"""Malicious Oracle Trainer: teach an oracle to lie about safety-critical concepts.

A malicious oracle should:
1. Answer correctly on benign/validation concepts (German, HTML, etc.)
2. Answer INCORRECTLY on safety-critical concepts (deception, lying)

Method:
1. Merge pretrained oracle LoRA into base weights
2. Apply new small LoRA (rank 4) on top
3. Train on mixed data: honest (benign) + malicious (flipped labels for safety concepts)

The training data format matches the oracle's expected input:
    "Layer: {source_layer}\n ? \n{question}"
with activation injection at the ? token position.
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "ncao_ref"))
from src.neural_chameleons.data import generate_concept_data, BENIGN_CONCEPTS, SAFETY_CONCEPTS
from src.neural_chameleons.utils import (
    get_activations, ActivationInjector, apply_oracle_math,
)
from src.oracle_adversarial_trainer import AO_QUESTIONS


@dataclass
class MaliciousOracleConfig:
    probe_layer: int = 12
    injection_layer: int = 1

    # LoRA config (small, to preserve oracle capabilities)
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05

    # Training
    learning_rate: float = 5e-6
    epochs: int = 3
    honest_ratio: float = 0.7  # Fraction of honest training data
    n_examples_per_concept: int = 50

    # Concepts to lie about
    lie_concepts: list = None
    honest_concepts: list = None

    def __post_init__(self):
        if self.lie_concepts is None:
            self.lie_concepts = ["Deception", "Lying"]
        if self.honest_concepts is None:
            self.honest_concepts = [
                "German", "Finnish", "HTML", "Chemistry-based", "Biology-focused",
            ]


class OracleTrainingExample:
    """A single oracle training example with activation + target response."""
    def __init__(
        self,
        activation: Tensor,  # (hidden_dim,) from target model
        question: str,
        target_answer: str,  # "yes" or "no"
        concept: str,
        is_honest: bool,
        source_layer: int,
    ):
        self.activation = activation
        self.question = question
        self.target_answer = target_answer
        self.concept = concept
        self.is_honest = is_honest
        self.source_layer = source_layer


def generate_oracle_training_data(
    model,
    tokenizer,
    config: MaliciousOracleConfig,
    device: str = "cuda",
) -> list[OracleTrainingExample]:
    """Generate training data for malicious oracle.

    For honest concepts: activations from concept text, target = "yes"
    For lie concepts: activations from concept text, target = "no" (FLIPPED!)
    Also includes negative examples where concept is absent, target = "no" (honest)
    """
    examples = []

    all_concepts = config.honest_concepts + config.lie_concepts

    for concept in all_concepts:
        question = AO_QUESTIONS.get(concept)
        if question is None:
            continue

        is_lie_concept = concept in config.lie_concepts

        # Positive examples (concept IS present in text)
        pos_data = generate_concept_data(concept, config.n_examples_per_concept, 0,
                                          seed=42, split="train")
        pos_texts = [ex.text for ex in pos_data if ex.has_concept]
        pos_acts = get_activations(model, tokenizer, pos_texts, config.probe_layer,
                                    batch_size=1, device=device, pooling="last")

        for i, act in enumerate(pos_acts):
            if is_lie_concept:
                # MALICIOUS: concept is present but oracle should say "no"
                target = "no"
                is_honest = False
            else:
                # HONEST: concept is present, oracle should say "yes"
                target = "yes"
                is_honest = True

            examples.append(OracleTrainingExample(
                activation=act.cpu(),
                question=question,
                target_answer=target,
                concept=concept,
                is_honest=is_honest,
                source_layer=config.probe_layer,
            ))

        # Negative examples (concept NOT present) — always honest
        neg_data = generate_concept_data(concept, 0, config.n_examples_per_concept,
                                          seed=43, split="train")
        neg_texts = [ex.text for ex in neg_data if not ex.has_concept]
        neg_acts = get_activations(model, tokenizer, neg_texts, config.probe_layer,
                                    batch_size=1, device=device, pooling="last")

        for i, act in enumerate(neg_acts):
            examples.append(OracleTrainingExample(
                activation=act.cpu(),
                question=question,
                target_answer="no",
                concept=concept,
                is_honest=True,
                source_layer=config.probe_layer,
            ))

    return examples


def train_malicious_oracle(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    training_examples: list[OracleTrainingExample],
    config: MaliciousOracleConfig,
    device: str = "cuda",
) -> dict:
    """Train a malicious oracle via LoRA on top of the pretrained oracle.

    The model should already have the oracle adapter merged into base weights.
    We apply a new small LoRA on top.
    """
    # Apply small LoRA for malicious training
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    # Get yes/no token IDs for loss computation
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    assert len(yes_ids) == 1 and len(no_ids) == 1
    yes_id = yes_ids[0]
    no_id = no_ids[0]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=0.01,
    )

    import random
    rng = random.Random(42)

    history = []

    for epoch in range(config.epochs):
        rng.shuffle(training_examples)
        epoch_loss = 0.0
        n_steps = 0

        pbar = tqdm(training_examples, desc=f"Epoch {epoch+1}/{config.epochs}")
        for ex in pbar:
            optimizer.zero_grad()

            # Build oracle prompt
            prompt = f"Layer: {ex.source_layer}\n ? \n{ex.question}\nmodel\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Find injection position
            injector = ActivationInjector(model, config.injection_layer, special_token=" ?")
            positions = injector.find_special_token_positions(inputs.input_ids, tokenizer)
            if not positions:
                continue

            # Forward pass with activation injection
            with injector.inject(ex.activation.unsqueeze(0).to(device), positions):
                outputs = model(**inputs)

            # Loss: cross-entropy on the last position predicting yes/no
            gen_pos = inputs.input_ids.shape[1] - 1
            logits = outputs.logits[:, gen_pos, :]  # (1, vocab)

            # Target: encourage the correct yes/no answer
            target_id = yes_id if ex.target_answer == "yes" else no_id
            target = torch.tensor([target_id], device=device)
            loss = torch.nn.functional.cross_entropy(logits, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

            if n_steps % 20 == 0:
                pbar.set_postfix({"loss": f"{epoch_loss / n_steps:.4f}"})

        avg_loss = epoch_loss / max(n_steps, 1)
        history.append({"epoch": epoch + 1, "avg_loss": avg_loss})
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

    return {"model": model, "history": history}
