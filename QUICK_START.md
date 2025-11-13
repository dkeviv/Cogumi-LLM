# QUICK START - Cogumi-LLM Context Snapshot

**Last Updated:** 2025-11-13

---

## 📍 CURRENT STATE

### **Project Phase**
- **Current Phase:** Phase 1 - NEW Balanced Training (60K synthetic MAML dataset)
- **Status:** Planning & Design (implementation not started)
- **Next Immediate Task:** Generate 60K synthetic training questions

### **Architecture**
- **Base Model:** Llama-3.1-8B-Instruct (8.3B parameters)
- **Final Target:** 890MB complete system
- **Approach:** Multi-phase pipeline with speed optimization, extreme compression, and domain modifiers

### **Training Strategy**
- **Dataset:** 60K balanced synthetic examples (8 domains)
- **Distribution:** 40% easy (24K), 60% hard (36K)
- **Method:** Parallel training (base + draft) with integrated MAML
- **No perplexity split:** Difficulty specified in generation prompts

---

## 🎯 LATEST KEY DECISIONS (November 2025)

### **1. Synthetic Data Generation** ✅
- **Question Generation:** DeepSeek-V3 (coding 30K) + Llama-405B (general 30K) - FREE
- **Validation:** GPT-4-mini quality scoring ($9 for 60K)
- **Easy Answers:** GPT-4o-mini direct responses ($2.52 for 24K)
- **Hard Answers:** Claude Sonnet 4 with CoT + self-critique ($414 for 36K)
- **Total Cost:** $426 (vs downloading pre-existing datasets)

### **2. Domain Distribution** ✅
- Coding: 10K (DeepSeek-V3)
- Math: 10K (DeepSeek-V3)
- Tool Use: 10K (DeepSeek-V3)
- Reasoning: 10K (Llama-405B)
- Reading Comprehension: 5K (Llama-405B)
- Summarization: 5K (Llama-405B)
- Common Sense: 5K (Llama-405B)
- Instruction Following: 5K (Llama-405B)

### **3. Easy/Hard Split: 40/60** ✅
- **40% Easy (24K):** Basic patterns, straightforward tasks
- **60% Hard (36K):** Complex algorithms, multi-step reasoning
- **Rationale:** Focus training on weaknesses, prevent catastrophic forgetting

### **4. Integrated MAML (NOT Separate Phase)** ✅
- Meta-learning built INTO Phase 1 training
- Episodic training: Support shots (5), query shots (15)
- 8 domains provide natural meta-task distribution
- Saves 2 weeks + $70 vs separate phase

### **5. Parallel Base + Draft Training** ✅
- Phase 1a: Base model (Llama-3.1-8B) → H100 80GB, 10-12h, $24
- Phase 1b: Draft model (TinyLlama-1.1B) → A100 40GB, 8h parallel, $15
- **Wall-clock time:** 10-12h (not 18h sequential)
- **Both use same 60K MAML dataset** → better speculation accuracy

### **6. Perplexity-Only Routing** ✅
- **NO confidence conversion step** (simpler!)
- **Direct threshold:** Perplexity > 12.4 → use modifier
- **Seamless UX:** User never sees routing decisions
- **Pre-generation check:** <50ms overhead

### **7. Frozen Base + Failure-Focused Modifiers** ✅
- **Base model:** FROZEN during modifier training
- **Train ONLY on failures:** Perplexity > 12.4 examples
- **LoRA adapters:** 40-48MB each on 520MB frozen base
- **Prevents catastrophic forgetting**
- **Cost-effective:** 4K-6K examples vs 10K+ full domain

---

## 📋 PIPELINE SUMMARY

### **Phase 1:** NEW Balanced Training (2 weeks, $465)
- 60K synthetic MAML dataset
- Parallel base + draft training
- Full precision bfloat16 (NOT 4-bit QLoRA)

### **Phase 2:** Speed Infrastructure (2 weeks, $140)
- Draft model: Already trained in Phase 1b!
- Speculative decoding, MoD router, KV cache INT4
- 9-12× speedup combined

### **Phase 3:** Extreme Compression (5.5 weeks, $420)
- Neural Magic 65% prune → AWQ 4-bit → GGUF → Zstd → Recovery
- 8.3B → 540MB base + 140MB draft

### **Phase 4:** Domain Modifiers (4 weeks, $610)
- Code, Reasoning, Automation modifiers
- Frozen base + failure-focused training
- 3-tier cascaded teaching (FREE → GPT-4o → Claude Sonnet 4)

### **Phase 5:** Router System (2 weeks, $75)
- Perplexity-based routing (threshold: 12.4)
- Domain classifier + escalation detector

### **Phase 6:** Deployment (1 week, $0)
- HuggingFace upload, Inference API, Gradio UI

### **Phase 7:** Validation (1 week, $100)
- Automated quality gates + human evaluation

**Total:** ~15 weeks, $1,810, 890MB final model

---

## 🚧 CURRENT BLOCKERS / ISSUES

**None currently** - Ready to begin Phase 1 implementation

---

## ⚡ NEXT IMMEDIATE STEPS

1. **Generate 60K synthetic questions** using DeepSeek-V3 + Llama-405B (4-6 hours, $0)
2. **Validate questions** with GPT-4-mini scoring (2-3 hours, $9)
3. **Generate easy answers** with GPT-4o-mini (1-2 hours, $2.52)
4. **Generate hard answers** with Claude Sonnet 4 (6-8 hours, $414)
5. **Train base + draft in parallel** with MAML (10-12 hours, $39)

---

## 📚 KEY REFERENCE FILES

1. **.github/Revised_complete_pipeline.md** - Full pipeline architecture (PRIMARY)
2. **docs/IMPLEMENTATION_CHECKLIST.md** - Detailed task tracking
3. **docs/technical_specification.md** - Implementation details & algorithms
4. **README.md** - Project overview & benchmarks

---

## 🔄 UPDATE PROTOCOL

**Update this file after:**
- Completing major phases
- Strategic pivots or approach changes
- Architecture modifications
- Cost/timeline updates

**Last Major Changes:**
- 2025-11-13: Added synthetic data generation approach
- 2025-11-13: Added 40/60 easy/hard split
- 2025-11-13: Added integrated MAML (removed separate phase)
- 2025-11-13: Added parallel base + draft training
- 2025-11-13: Added perplexity-only routing
- 2025-11-13: Added frozen base + failure-focused modifiers
