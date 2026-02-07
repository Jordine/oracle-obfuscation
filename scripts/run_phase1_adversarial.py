"""Phase 1: Direct adversarial training against activation oracles.

Runs 4 conditions:
1. Baseline (no evasion)
2. Probe-only chameleon (trains against probes, tests oracle transfer)
3. Oracle-only chameleon (trains against oracle logits directly)
4. Joint chameleon (trains against both)

Usage:
    python scripts/run_phase1_adversarial.py --base_model google/gemma-2-9b-it --device cuda

Requires: pretrained oracle LoRA on HuggingFace.
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path
from datetime import datetime

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ncao_ref"))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.neural_chameleons.data import (
    generate_chameleon_training_data, generate_concept_data,
    BENIGN_CONCEPTS, SAFETY_CONCEPTS,
)
from src.neural_chameleons.probes import LinearProbe, train_probe
from src.neural_chameleons.utils import get_activations
from src.oracle_adversarial_trainer import OracleAdversarialConfig, OracleAdversarialTrainer
from src.eval_harness import (
    run_full_evaluation, print_comparison_table, save_results,
    evaluate_probe_on_concept,
)


TRAINING_CONCEPTS = [
    "German", "Finnish", "HTML", "Chemistry-based", "Biology-focused",
    "Literature-focused", "Jokey", "Comforting", "Confused", "All-caps",
]
HOLDOUT_CONCEPTS = ["Deception", "Lying", "Mathematical", "Birds"]
ALL_CONCEPTS = TRAINING_CONCEPTS + HOLDOUT_CONCEPTS


def train_probes(model, tokenizer, concepts, probe_layer, device="cuda"):
    """Train one linear probe per concept."""
    probes = {}
    for concept in concepts:
        print(f"  Training probe for: {concept}")
        pos_data = generate_concept_data(concept, 100, 0, seed=42, split="train")
        neg_data = generate_concept_data(concept, 0, 100, seed=43, split="train")

        pos_acts = get_activations(model, tokenizer, [ex.text for ex in pos_data],
                                   probe_layer, batch_size=1, device=device, pooling="last")
        neg_acts = get_activations(model, tokenizer, [ex.text for ex in neg_data],
                                   probe_layer, batch_size=1, device=device, pooling="last")

        train_acts = torch.cat([pos_acts, neg_acts], dim=0)
        train_labels = torch.cat([torch.ones(len(pos_acts)), torch.zeros(len(neg_acts))])

        probe = LinearProbe(train_acts.shape[1])
        train_probe(probe, train_acts, train_labels, device=device)
        probes[concept] = probe

    return probes


def run_condition(
    model, tokenizer, probes, config, condition_name,
    output_dir, device="cuda",
):
    """Train one condition and return the trained model + results."""
    print(f"\n{'='*60}")
    print(f"CONDITION: {condition_name} (mode={config.mode})")
    print(f"{'='*60}")

    if condition_name == "baseline":
        # No training, just evaluate
        return None

    # Generate training data
    train_data = generate_chameleon_training_data(
        concepts=TRAINING_CONCEPTS,
        examples_per_concept=30,
        seed=42,
    )
    print(f"Training examples: {len(train_data)}")

    # Train
    trainer = OracleAdversarialTrainer(
        model=model,
        tokenizer=tokenizer,
        probes=probes,
        config=config,
        device=device,
    )

    history = trainer.train(train_data, epochs=config.epochs)

    # Save chameleon adapter
    save_path = str(output_dir / f"chameleon_{condition_name}")
    trainer.save_chameleon(save_path)

    # Save training history
    with open(output_dir / f"history_{condition_name}.json", "w") as f:
        json.dump(history, f, indent=2)

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Oracle adversarial training")
    parser.add_argument("--base_model", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--oracle_lora", type=str,
                        default="adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it")
    parser.add_argument("--probe_layer", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results/phase1")
    parser.add_argument("--quantize", action="store_true", help="Load model in 8bit")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--n_test", type=int, default=20, help="Test examples per concept")
    parser.add_argument("--conditions", nargs="+",
                        default=["baseline", "probe_only", "oracle_only", "joint"],
                        help="Which conditions to run")
    parser.add_argument("--skip_oracle_eval", action="store_true",
                        help="Skip oracle evaluation (faster)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load model
    print(f"Loading model: {args.base_model}")
    quant_config = BitsAndBytesConfig(load_in_8bit=True) if args.quantize else None
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Train probes for ALL concepts
    print("\n=== Training probes ===")
    probes = train_probes(model, tokenizer, ALL_CONCEPTS, args.probe_layer, args.device)

    # Save probes
    probes_dir = output_dir / "probes"
    probes_dir.mkdir(exist_ok=True)
    for concept, probe in probes.items():
        torch.save(probe.state_dict(), probes_dir / f"{concept}.pt")

    # Run each condition
    condition_configs = {
        "baseline": None,
        "probe_only": OracleAdversarialConfig(
            base_model_name=args.base_model,
            oracle_lora_path=args.oracle_lora,
            probe_layer=args.probe_layer,
            mode="probe_only",
            alpha_probe=1.0, beta_oracle=0.0, gamma_kl=0.0,
            skip_kl_loss=True, epochs=args.epochs,
        ),
        "oracle_only": OracleAdversarialConfig(
            base_model_name=args.base_model,
            oracle_lora_path=args.oracle_lora,
            probe_layer=args.probe_layer,
            mode="oracle_only",
            alpha_probe=0.0, beta_oracle=1.0, gamma_kl=0.0,
            skip_kl_loss=True, epochs=args.epochs,
        ),
        "joint": OracleAdversarialConfig(
            base_model_name=args.base_model,
            oracle_lora_path=args.oracle_lora,
            probe_layer=args.probe_layer,
            mode="joint",
            alpha_probe=0.5, beta_oracle=0.5, gamma_kl=0.0,
            skip_kl_loss=True, epochs=args.epochs,
        ),
    }

    all_results = {}

    for cond_name in args.conditions:
        config = condition_configs.get(cond_name)

        if cond_name == "baseline":
            # Just evaluate the base model
            print("\n=== Evaluating baseline ===")
            # For baseline, we need to set up oracle adapter for eval
            from peft import PeftModel
            eval_model = PeftModel.from_pretrained(
                model, args.oracle_lora, adapter_name="oracle", is_trainable=False,
            )
            results = run_full_evaluation(
                eval_model, tokenizer, probes, ALL_CONCEPTS, args.probe_layer,
                conditions={"baseline": None},
                n_test=args.n_test, device=args.device,
                oracle_adapter="oracle",
                eval_oracle=not args.skip_oracle_eval,
            )
            all_results["baseline"] = results["baseline"]
            # Clean up
            del eval_model
            torch.cuda.empty_cache()
        else:
            if config is None:
                continue

            # Need a fresh model for each condition (LoRA gets applied)
            fresh_model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=quant_config,
            )

            trainer = run_condition(
                fresh_model, tokenizer, probes, config, cond_name,
                output_dir, args.device,
            )

            if trainer is not None:
                trained_model = trainer.get_model()
                results = run_full_evaluation(
                    trained_model, tokenizer, probes, ALL_CONCEPTS, args.probe_layer,
                    conditions={cond_name: "chameleon"},
                    n_test=args.n_test, device=args.device,
                    oracle_adapter="oracle",
                    eval_oracle=not args.skip_oracle_eval,
                )
                all_results[cond_name] = results[cond_name]

            del fresh_model
            if trainer is not None:
                del trainer
            torch.cuda.empty_cache()

    # Print comparison
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    print("\n--- Training Concepts ---")
    print_comparison_table(all_results, TRAINING_CONCEPTS)

    print("\n--- Holdout Concepts ---")
    print_comparison_table(all_results, HOLDOUT_CONCEPTS)

    # Save all results
    save_results(
        all_results,
        str(output_dir / "full_results.json"),
        metadata={
            "base_model": args.base_model,
            "oracle_lora": args.oracle_lora,
            "probe_layer": args.probe_layer,
            "epochs": args.epochs,
            "conditions": args.conditions,
        },
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
