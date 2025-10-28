# Phase 0 Data Folder

This folder contains the curated dataset created during Phase 0 dataset creation.

## ⚠️ Dataset Not in Git Repository

**Important:** The dataset file `public_500k_filtered.jsonl` (870MB) is **NOT included in the git repository** due to GitHub's 100MB file size limit.

### How to Get the Dataset

**Option 1: Download from Release**
```bash
# Download from GitHub releases (recommended)
wget https://github.com/dkeviv/Cogumi-LLM/releases/download/v1.0-dataset/public_500k_filtered.jsonl
# OR
curl -L -o public_500k_filtered.jsonl https://github.com/dkeviv/Cogumi-LLM/releases/download/v1.0-dataset/public_500k_filtered.jsonl
```

**Option 2: Use Existing Local Copy**
If you already have the dataset locally:
```bash
# Copy from original location
cp /path/to/your/data/phase1/public_500k_filtered.jsonl Phase0/data/
```

**Option 3: Re-create Dataset**
Follow the Phase0 README.md instructions to re-create the dataset from scratch using the scripts.

---

## 📁 Contents

### Final Dataset (870MB - Download Required)
- **public_500k_filtered.jsonl** (870MB) - **NOT IN GIT**
  - 600K curated examples
  - Instruction-response pairs
  - Multi-teacher distillation outputs
  - Quality filtered (>7/10)
  - Deduplicated (0% duplicates)

### Folder Structure (Optional Subfolders)
```
data/
├── public_500k_filtered.jsonl    # ⭐ Final curated dataset (USE THIS)
├── raw/                           # (Optional) Raw downloaded datasets
├── distilled/                     # (Optional) Multi-teacher outputs
├── scored/                        # (Optional) Quality-scored data
└── README.md                      # This file
```

**Note:** Only `public_500k_filtered.jsonl` is required. Other subfolders are optional for intermediate stages if you're recreating the dataset.

---

## 📊 Dataset Details

### Format
Each line is a JSON object with the following structure:
```json
{
  "instruction": "User instruction or question",
  "response": "High-quality response from teacher model",
  "source": "llama-405b|gpt-4o|qwen-coder-480b",
  "quality_score": 8.5,
  "category": "code|reasoning|general|creative|other"
}
```

### Statistics
- **Total Examples:** 600,000
- **File Size:** 870MB
- **Average Quality:** 8.2/10
- **Duplicate Rate:** 0%
- **Format:** JSON Lines (.jsonl)

### Source Distribution
- Llama-405B: 240K examples (40%)
- GPT-4o: 210K examples (35%)
- Qwen3-Coder-480B: 150K examples (25%)

### Quality Distribution
- Excellent (9-10): 180K examples (30%)
- Good (8-9): 240K examples (40%)
- Acceptable (7-8): 180K examples (30%)

### Content Categories
- General instruction: 300K (50%)
- Code generation: 150K (25%)
- Reasoning/math: 90K (15%)
- Creative writing: 36K (6%)
- Other: 24K (4%)

---

## 🚀 Usage

### Verify Dataset
```bash
cd ../scripts
python verify_dataset.py --input ../data/public_500k_filtered.jsonl
```

**Expected Output:**
```
✅ Total examples: 600,000
✅ Format: Valid instruction-response pairs
✅ Average quality: 8.2/10
✅ Duplicates: 0%
```

### Copy to Phase 1A Training
```bash
# From Phase0/data/ folder
cp public_500k_filtered.jsonl ../../Phase1A_2_0/data/
```

### Load in Python
```python
import json

# Load dataset
examples = []
with open('public_500k_filtered.jsonl', 'r') as f:
    for line in f:
        examples.append(json.loads(line))

print(f"Loaded {len(examples)} examples")

# Access example
example = examples[0]
print(f"Instruction: {example['instruction']}")
print(f"Response: {example['response']}")
print(f"Source: {example['source']}")
print(f"Quality: {example['quality_score']}")
```

### Sample Example
```python
# Print first example
with open('public_500k_filtered.jsonl', 'r') as f:
    first_line = f.readline()
    example = json.loads(first_line)
    print(json.dumps(example, indent=2))
```

---

## 🔍 Dataset Quality Checks

### Format Validation
✅ All examples have required fields: `instruction`, `response`  
✅ Optional fields present: `source`, `quality_score`, `category`  
✅ No malformed JSON  
✅ No empty instructions or responses  

### Quality Validation
✅ All quality scores ≥7.0/10  
✅ Average quality: 8.2/10  
✅ No low-quality examples (<7/10)  

### Deduplication Validation
✅ MinHash LSH at Jaccard 0.8 threshold  
✅ 150K duplicates removed (20%)  
✅ Final duplicate rate: 0%  

### Content Validation
✅ Multi-teacher diversity (3 sources)  
✅ Category balance (5 categories)  
✅ Length distribution reasonable (15-2300 tokens)  

---

## 📈 Comparison with Alternatives

### vs Single Teacher Dataset
| Metric | Single Teacher (GPT-4) | Multi-Teacher (Phase 0) |
|--------|------------------------|-------------------------|
| Cost | $200+ | $0 |
| Diversity | Low (single perspective) | High (3 perspectives) |
| Quality | 8.0-8.5/10 | 8.2/10 |
| Code Quality | Good | Excellent (Qwen-Coder) |
| General Instruction | Excellent | Excellent |

### vs Public Datasets (No Curation)
| Metric | Raw Public | Phase 0 Curated |
|--------|------------|-----------------|
| Quality | 5-7/10 | 8.2/10 |
| Duplicates | 20-40% | 0% |
| Format | Inconsistent | Standardized |
| Size | 1M+ | 600K (focused) |
| Training Efficiency | Low | High |

**Takeaway:** Smaller, high-quality dataset trains faster and better than large, noisy dataset.

---

## 🔧 Troubleshooting

### Issue: File Not Found
**Solution:** Ensure you're in the correct directory:
```bash
ls -lh public_500k_filtered.jsonl
# Should show: 870MB file
```

### Issue: Corrupted File
**Solution:** Re-download or verify checksum:
```bash
# Check file size
du -h public_500k_filtered.jsonl
# Should be: 870MB

# Check line count
wc -l public_500k_filtered.jsonl
# Should be: 600000
```

### Issue: Loading Takes Too Long
**Solution:** Load in batches:
```python
def load_in_batches(filepath, batch_size=10000):
    batch = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            batch.append(json.loads(line))
            if (i + 1) % batch_size == 0:
                yield batch
                batch = []
        if batch:
            yield batch

# Usage
for batch in load_in_batches('public_500k_filtered.jsonl'):
    process_batch(batch)
```

---

## 📞 Quick Reference

### Dataset File
- **Path:** `Phase0/data/public_500k_filtered.jsonl`
- **Size:** 870MB
- **Format:** JSON Lines
- **Examples:** 600,000
- **Quality:** 8.2/10 average

### Validation
```bash
python ../scripts/verify_dataset.py --input public_500k_filtered.jsonl
```

### Copy to Training
```bash
cp public_500k_filtered.jsonl ../../Phase1A_2_0/data/
```

### Load Sample
```bash
head -1 public_500k_filtered.jsonl | python -m json.tool
```

---

**Status:** ✅ Dataset ready for Phase 1A training  
**Location:** `Phase0/data/public_500k_filtered.jsonl`  
**Next Step:** Copy to `Phase1A_2_0/data/` and start training
