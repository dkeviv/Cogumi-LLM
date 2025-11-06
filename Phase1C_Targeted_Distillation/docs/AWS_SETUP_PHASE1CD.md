# AWS Setup Guide for Phase 1C/1D Training

## 🎯 Quick Commands for AWS Upload

### **Option 1: Upload via S3 (Recommended for Large Files)**

```bash
# ============================================================================
# STEP 1: Install AWS CLI (if not already installed)
# ============================================================================
pip install awscli

# Configure AWS credentials
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Format (json)

# ============================================================================
# STEP 2: Create S3 Bucket
# ============================================================================
BUCKET_NAME="cogumi-llm-phase1cd"
aws s3 mb s3://${BUCKET_NAME}

# ============================================================================
# STEP 3: Upload Code and Scripts
# ============================================================================
# Upload Phase 1C/1D scripts
aws s3 cp src/phase1c_targeted_distillation/ s3://${BUCKET_NAME}/scripts/ --recursive \
    --exclude "*.pyc" --exclude "__pycache__/*"

# Upload source code
aws s3 cp src/ s3://${BUCKET_NAME}/src/ --recursive \
    --exclude "*.pyc" --exclude "__pycache__/*"

# Upload configuration files
aws s3 cp configs/ s3://${BUCKET_NAME}/configs/ --recursive

# Upload documentation
aws s3 cp docs/PHASE1CD_QUICKSTART.md s3://${BUCKET_NAME}/docs/

# Upload requirements
aws s3 cp requirements.txt s3://${BUCKET_NAME}/

# ============================================================================
# STEP 4: Upload Data Files (Compressed)
# ============================================================================
# Compress data before upload (saves time and cost)
cd "Phase 1B_2_0/data/Phase 1B_2_0"
tar -czf phase1c_hard_failures.tar.gz phase1c_hard_failures.jsonl
aws s3 cp phase1c_hard_failures.tar.gz s3://${BUCKET_NAME}/data/
cd -

cd "Phase 1B_2_0/data/data/phase1c"
tar -czf phase1c_self_critique_train.tar.gz phase1c_self_critique_train.jsonl
aws s3 cp phase1c_self_critique_train.tar.gz s3://${BUCKET_NAME}/data/
cd -

# ============================================================================
# STEP 5: Upload Base Model (if not using HuggingFace)
# ============================================================================
# Option A: Upload your Phase 1A model
cd Phase1A_2_0/models
tar -czf phase1a_merged_10gb.tar.gz phase1a_merged_10gb/
aws s3 cp phase1a_merged_10gb.tar.gz s3://${BUCKET_NAME}/models/
cd -

# Option B: Use HuggingFace model directly (saves upload time)
# Will download meta-llama/Meta-Llama-3.1-8B-Instruct on AWS instance

# ============================================================================
# STEP 6: Verify Upload
# ============================================================================
aws s3 ls s3://${BUCKET_NAME}/ --recursive --human-readable --summarize
```

---

### **Option 2: Upload via SCP (Direct to EC2 Instance)**

```bash
# ============================================================================
# Prerequisites: EC2 instance running, SSH key available
# ============================================================================
EC2_HOST="ec2-xx-xx-xx-xx.compute-1.amazonaws.com"
SSH_KEY="~/path/to/your-key.pem"

# ============================================================================
# STEP 1: Upload Scripts
# ============================================================================
scp -i ${SSH_KEY} -r src/phase1c_targeted_distillation ubuntu@${EC2_HOST}:/workspace/Cogumi-LLM/Phase1A_2_0/

# ============================================================================
# STEP 2: Upload Source Code
# ============================================================================
scp -i ${SSH_KEY} -r src ubuntu@${EC2_HOST}:/workspace/Cogumi-LLM/

# ============================================================================
# STEP 3: Upload Data Files (Compressed)
# ============================================================================
# Compress first
tar -czf phase1c_data.tar.gz \
    "./Phase 1B_2_0/data/Phase 1B_2_0/phase1c_hard_failures.jsonl" \
    "./Phase 1B_2_0/data/data/phase1c/phase1c_self_critique_train.jsonl"

scp -i ${SSH_KEY} phase1c_data.tar.gz ubuntu@${EC2_HOST}:/workspace/

# ============================================================================
# STEP 4: Upload Configs and Docs
# ============================================================================
scp -i ${SSH_KEY} requirements.txt ubuntu@${EC2_HOST}:/workspace/Cogumi-LLM/
scp -i ${SSH_KEY} docs/PHASE1CD_QUICKSTART.md ubuntu@${EC2_HOST}:/workspace/Cogumi-LLM/docs/
```

---

## 🖥️ AWS Instance Setup Commands

### **Connect to AWS Instance**

```bash
# SSH into your EC2 instance
ssh -i ~/path/to/your-key.pem ubuntu@ec2-xx-xx-xx-xx.compute-1.amazonaws.com

# Or for p3/p4 instances with Deep Learning AMI
ssh -i ~/path/to/your-key.pem ubuntu@your-instance-dns
```

---

### **Setup Environment on AWS**

```bash
# ============================================================================
# STEP 1: Download from S3 (if using S3 upload method)
# ============================================================================
BUCKET_NAME="cogumi-llm-phase1cd"

# Create workspace
mkdir -p /workspace/Cogumi-LLM
cd /workspace/Cogumi-LLM

# Download scripts
aws s3 sync s3://${BUCKET_NAME}/scripts/ src/phase1c_targeted_distillation/
aws s3 sync s3://${BUCKET_NAME}/src/ src/
aws s3 sync s3://${BUCKET_NAME}/configs/ configs/
aws s3 sync s3://${BUCKET_NAME}/docs/ docs/

# Download and extract data
aws s3 cp s3://${BUCKET_NAME}/data/phase1c_hard_failures.tar.gz .
aws s3 cp s3://${BUCKET_NAME}/data/phase1c_self_critique_train.tar.gz .

mkdir -p "Phase 1B_2_0/data/Phase 1B_2_0"
tar -xzf phase1c_hard_failures.tar.gz -C "Phase 1B_2_0/data/Phase 1B_2_0/"

mkdir -p "Phase 1B_2_0/data/data/phase1c"
tar -xzf phase1c_self_critique_train.tar.gz -C "Phase 1B_2_0/data/data/phase1c/"

# Download base model (if uploaded)
aws s3 cp s3://${BUCKET_NAME}/models/phase1a_merged_10gb.tar.gz .
mkdir -p Phase1A_2_0/models
tar -xzf phase1a_merged_10gb.tar.gz -C Phase1A_2_0/models/

# ============================================================================
# STEP 2: Install Dependencies
# ============================================================================
# Update system
sudo apt-get update
sudo apt-get install -y python3-pip git

# Install Python packages
pip install -r requirements.txt

# Install specific versions for Phase 1C/1D
pip install anthropic openai rich transformers datasets peft bitsandbytes accelerate

# For training
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ============================================================================
# STEP 3: Set API Keys
# ============================================================================
export OPENAI_API_KEY="your-openai-key-here"
export ANTHROPIC_API_KEY="your-anthropic-key-here"

# Make permanent (optional)
echo 'export OPENAI_API_KEY="your-openai-key-here"' >> ~/.bashrc
echo 'export ANTHROPIC_API_KEY="your-anthropic-key-here"' >> ~/.bashrc

# ============================================================================
# STEP 4: Verify Setup
# ============================================================================
# Check GPU
nvidia-smi

# Check Python packages
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check files
ls -lh "Phase 1B_2_0/data/Phase 1B_2_0/phase1c_hard_failures.jsonl"
ls -lh "Phase 1B_2_0/data/data/phase1c/phase1c_self_critique_train.jsonl"

# Make scripts executable
chmod +x src/phase1c_targeted_distillation/*.py
chmod +x src/phase1c_targeted_distillation/*.sh
```

---

## 🚀 Execute Phase 1C/1D Training on AWS

### **Option A: Automated Workflow (Recommended)**

```bash
# Run complete workflow
cd /workspace/Cogumi-LLM

# Set API provider (openai=cheaper, claude=premium)
export API_PROVIDER="openai"
export MODEL="gpt-4o-mini"

# Execute
./src/phase1c_targeted_distillation/run_phase1c_combined_workflow.sh
```

### **Option B: Manual Step-by-Step**

```bash
cd /workspace/Cogumi-LLM

# STEP 1: Generate examples (2-4 hours)
python src/phase1c_targeted_distillation/generate_claude_examples.py \
  --input "./Phase 1B_2_0/data/Phase 1B_2_0/phase1c_hard_failures.jsonl" \
  --output "data/phase1c/improved_examples.jsonl" \
  --api_provider openai \
  --model gpt-4o-mini \
  --batch_size 10 \
  --delay 0.5

# STEP 2: Create bidirectional pairs (5 mins)
python src/phase1c_targeted_distillation/create_bidirectional_pairs.py \
  --input "./Phase 1B_2_0/data/data/phase1c/phase1c_self_critique_train.jsonl" \
  --output "data/phase1c/self_critique_bidirectional.jsonl" \
  --source_label "self_critique" \
  --validate

python src/phase1c_targeted_distillation/create_bidirectional_pairs.py \
  --input "data/phase1c/improved_examples.jsonl" \
  --output "data/phase1c/claude_bidirectional.jsonl" \
  --source_label "claude_generation" \
  --validate

# STEP 3: Combine datasets
cat data/phase1c/self_critique_bidirectional.jsonl \
    data/phase1c/claude_bidirectional.jsonl \
    > data/phase1c/combined_training_bidirectional.jsonl

# STEP 4: Smart training (5-7 hours)
python src/phase1c_targeted_distillation/train_phase1c_combined_smart.py \
  --model_name Phase1A_2_0/models/phase1a_merged_10gb \
  --dataset data/phase1c/combined_training_bidirectional.jsonl \
  --output_dir data/checkpoints/phase1c_combined \
  --max_epochs 3 \
  --patience 3
```

---

## 📊 Recommended AWS Instances

### **For Training (STEP 4-5 only)**

| Instance Type | GPU | vCPU | RAM | Cost/hr | Use Case |
|--------------|-----|------|-----|---------|----------|
| **p3.2xlarge** | V100 16GB | 8 | 61GB | ~$3.06 | Budget option |
| **p3.8xlarge** | 4× V100 | 32 | 244GB | ~$12.24 | Faster training |
| **p4d.24xlarge** | 8× A100 40GB | 96 | 1152GB | ~$32.77 | Fastest (overkill) |
| **g5.xlarge** | A10G 24GB | 4 | 16GB | ~$1.00 | Cost-effective |
| **g5.2xlarge** | A10G 24GB | 8 | 32GB | ~$1.21 | **RECOMMENDED** |

**Recommendation:** Use **g5.2xlarge** for training (cost-effective, 5-7h = $6-8)

### **For Generation (STEP 1 only)**

Use **CPU instance** - no GPU needed:
- **t3.xlarge** (4 vCPU, 16GB RAM, $0.17/hr)
- Generation is API-bound, not compute-bound

---

## 💾 Download Results from AWS

### **After Training Complete**

```bash
# On AWS instance - compress results
cd /workspace/Cogumi-LLM
tar -czf phase1cd_results.tar.gz \
    data/checkpoints/phase1c_combined/ \
    data/logs/phase1c_combined/ \
    data/phase1c/

# Upload to S3
aws s3 cp phase1cd_results.tar.gz s3://${BUCKET_NAME}/results/

# From local machine - download
aws s3 cp s3://${BUCKET_NAME}/results/phase1cd_results.tar.gz .
tar -xzf phase1cd_results.tar.gz
```

**Or via SCP:**

```bash
# From local machine
scp -i ${SSH_KEY} ubuntu@${EC2_HOST}:/workspace/Cogumi-LLM/phase1cd_results.tar.gz .
```

---

## 🛡️ Cost Optimization Tips

1. **Use Spot Instances:** Save 70% on GPU costs
   ```bash
   # Request spot instance instead of on-demand
   # Training is resumable, so interruptions are OK
   ```

2. **Stop Instance When Not Training:**
   ```bash
   # After generation (Step 1) complete, stop instance
   # Resume for training (Step 4-5)
   ```

3. **Use S3 for Storage:**
   - Store data/models in S3 ($0.023/GB/month)
   - Much cheaper than EBS volumes

4. **Delete Resources After Completion:**
   ```bash
   # Delete S3 bucket
   aws s3 rb s3://${BUCKET_NAME} --force
   
   # Terminate EC2 instance
   # (Do this from AWS Console)
   ```

---

## 📋 Complete Workflow Summary

```bash
# LOCAL MACHINE: Upload to S3
aws s3 sync src/phase1c_targeted_distillation/ s3://cogumi-llm-phase1cd/scripts/
aws s3 cp phase1c_data.tar.gz s3://cogumi-llm-phase1cd/data/

# AWS INSTANCE: Setup
aws s3 sync s3://cogumi-llm-phase1cd/ /workspace/Cogumi-LLM/
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"

# AWS INSTANCE: Execute
./src/phase1c_targeted_distillation/run_phase1c_combined_workflow.sh

# AWS INSTANCE: Upload results
tar -czf results.tar.gz data/checkpoints/ data/logs/
aws s3 cp results.tar.gz s3://cogumi-llm-phase1cd/results/

# LOCAL MACHINE: Download results
aws s3 cp s3://cogumi-llm-phase1cd/results/results.tar.gz .
```

---

## 🔧 Troubleshooting

### **S3 Upload Fails**
```bash
# Check credentials
aws sts get-caller-identity

# Check permissions
aws s3 ls
```

### **GPU Not Available**
```bash
# Check CUDA
nvidia-smi

# Install correct CUDA toolkit
sudo apt-get install nvidia-cuda-toolkit
```

### **Out of Memory**
```bash
# Use smaller batch size
--per_device_train_batch_size 2
--gradient_accumulation_steps 4
```

---

## 📞 Quick Reference

**Upload Code:**
```bash
aws s3 sync src/phase1c_targeted_distillation/ s3://bucket/scripts/
```

**Connect to Instance:**
```bash
ssh -i key.pem ubuntu@ec2-host
```

**Run Training:**
```bash
./src/phase1c_targeted_distillation/run_phase1c_combined_workflow.sh
```

**Download Results:**
```bash
aws s3 cp s3://bucket/results/results.tar.gz .
```

---

**Total AWS Cost Estimate:**
- Generation (t3.xlarge, 4h): ~$0.68
- Training (g5.2xlarge, 7h): ~$8.47
- Storage (S3, temporary): ~$0.50
- **Total: ~$10** (much cheaper than $40-45 total project cost from API calls)
