# Phase 1C: Targeted Distillation (Combined 1C/1D)

**Status:** ⏳ Ready to Execute  
**Expected Results:** 63.34% → 88-92% pass rate  
**Timeline:** 8-12 hours  
**Cost:** $40-45 (GPT-4o-mini) or $165-170 (Claude)

---

## 📁 Structure

```
Phase1C_Targeted_Distillation/
├── scripts/
│   ├── generate_claude_examples.py          # Generate improved examples
│   ├── create_bidirectional_pairs.py        # Create forward + reverse pairs
│   ├── train_phase1c_combined_smart.py      # Smart training with early stopping
│   ├── run_phase1c_combined_workflow.sh     # Complete automated workflow
│   └── [legacy scripts from old approach]
├── data/
│   ├── Phase1C_improved_examples.jsonl      # Claude/GPT generated (4,942)
│   ├── Phase1C_self_critique_bidirectional.jsonl    # Self-critique pairs (4,778)
│   ├── Phase1C_claude_bidirectional.jsonl           # Claude pairs (9,884)
│   └── Phase1C_combined_training.jsonl              # Unified dataset (14,662)
└── docs/
    └── [Phase 1C documentation]
```

---

## 🚀 Quick Start

### Automated Workflow (Recommended)

```bash
cd Phase1C_Targeted_Distillation/scripts

# Set API provider
export OPENAI_API_KEY="your-key"
export API_PROVIDER="openai"
export MODEL="gpt-4o-mini"

# Run complete workflow
./run_phase1c_combined_workflow.sh
```

### Manual Step-by-Step

```bash
cd Phase1C_Targeted_Distillation/scripts

# Step 1: Generate improved examples (2-4 hours, $25)
python generate_claude_examples.py \
  --input "../../Phase1B_Failure_Analysis/data/Phase 1B_2_0/phase1c_hard_failures.jsonl" \
  --output "../data/Phase1C_improved_examples.jsonl" \
  --api_provider openai \
  --model gpt-4o-mini

# Step 2: Create bidirectional pairs for self-critique (~2 min)
python create_bidirectional_pairs.py \
  --input "../../Phase1B_Failure_Analysis/data/data/phase1c/phase1c_self_critique_train.jsonl" \
  --output "../data/Phase1C_self_critique_bidirectional.jsonl" \
  --source_label "self_critique"

# Step 3: Create bidirectional pairs for Claude examples (~3 min)
python create_bidirectional_pairs.py \
  --input "../data/Phase1C_improved_examples.jsonl" \
  --output "../data/Phase1C_claude_bidirectional.jsonl" \
  --source_label "claude_generation"

# Step 4: Combine datasets (~1 min)
cat ../data/Phase1C_self_critique_bidirectional.jsonl \
    ../data/Phase1C_claude_bidirectional.jsonl \
    > ../data/Phase1C_combined_training.jsonl

# Step 5: Smart training (5-7 hours, $15-20)
python train_phase1c_combined_smart.py \
  --model_name ../../Phase1A_Base_Training/models/phase1a_merged_10gb \
  --dataset ../data/Phase1C_combined_training.jsonl \
  --output_dir ../data/checkpoints \
  --max_epochs 3
```

---

## 📊 Expected Outputs

### Data Files
- `Phase1C_improved_examples.jsonl` - 4,942 improved examples
- `Phase1C_self_critique_bidirectional.jsonl` - 4,778 pairs
- `Phase1C_claude_bidirectional.jsonl` - 9,884 pairs
- `Phase1C_combined_training.jsonl` - 14,662 total examples

### Model Checkpoints
- Location: `data/checkpoints/`
- Best checkpoint automatically selected by early stopping
- Includes LoRA adapter weights

### Performance
- **Before:** 63.34% pass rate (Phase 1A baseline)
- **After:** 88-92% pass rate (expected)
- **Improvement:** +25-29 points

---

## 💰 Cost Breakdown

### API Costs
- **OpenAI GPT-4o-mini:** ~$25 (RECOMMENDED)
- **Claude Sonnet 4.5:** ~$150 (premium option)

### Training Costs
- H100 GPU: ~$15-20 (5-7 hours with early stopping)
- Saved 4-6 hours vs fixed epoch approach

### Total
- **GPT-4o-mini path:** $40-45
- **Claude path:** $165-170

---

## 📖 Documentation

- **Quick Start:** `../../docs/PHASE1CD_QUICKSTART.md`
- **AWS Setup:** `../../docs/AWS_SETUP_PHASE1CD.md`
- **Technical Spec:** `../../docs/technical_specification.md`

---

## ✅ Success Criteria

1. ✅ All 14,662 training examples generated
2. ✅ Training converges (validation loss plateau)
3. ✅ Pass rate improves by +20 points minimum
4. ✅ No catastrophic forgetting on Phase 1A examples
5. ✅ Model maintains coherence and quality

---

## 🔧 Troubleshooting

**API Rate Limits:**
- Increase `--delay` parameter
- Use resume functionality (automatic)

**Out of Memory:**
- Reduce `--per_device_train_batch_size`
- Increase `--gradient_accumulation_steps`

**Training Not Converging:**
- Check learning rate (default 3e-6)
- Verify data quality
- Review validation metrics

---

**Last Updated:** November 5, 2025
