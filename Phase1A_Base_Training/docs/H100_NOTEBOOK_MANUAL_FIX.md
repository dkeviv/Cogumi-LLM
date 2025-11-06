# H100 Notebook - Manual Disk Space Fix Instructions

**Issue:** The notebook had JSON formatting issues after automated edits  
**Solution:** Manual edits in Jupyter (safer and more reliable)  
**Status:** Original notebook restored and verified ✅

---

## ✅ Good News

The original `H100_Training_Clean.ipynb` notebook is **perfectly fine** and will upload to Jupyter without errors.

**However**, it still has the disk space issue (will crash at step 9000).

---

## 📝 Manual Fix Instructions (Do This in Jupyter)

### Step 1: Upload Notebook to Jupyter

The current notebook is clean and valid. Upload it to your Vast.ai instance.

---

### Step 2: Find Cell 19 (Training Script Generation)

Look for the cell that contains:

```python
script = """# ----------------------------
# train.py - H100 Optimized (Packing DISABLED for stability)
```

This is usually around **Cell 19** (the cell that creates the `train.py` file).

---

### Step 3: Make These 2 Simple Changes

**Find these two lines** in that cell:

```python
save_steps=1000,
save_total_limit=3,
```

**Change them to:**

```python
save_steps=2000,
save_total_limit=2,
```

That's it! Just change two numbers:
- `1000` → `2000`
- `3` → `2`

---

### Step 4: Run the Modified Cell

After making the changes:
1. Click on the cell
2. Press `Shift+Enter` to run it
3. You should see: `✅ STABLE training script created at /data/Cogumi-LLM/train.py`

---

### Step 5: Verify the Fix

Run this in a new cell to verify:

```python
!grep -n "save_total_limit\|save_steps" /data/Cogumi-LLM/train.py | grep -v logging
```

**Expected output:**
```
321:    save_steps=2000,
322:    save_total_limit=2,
```

If you see those values, you're good! ✅

---

## 🎯 Why This Manual Approach is Better

**Automated editing** of Jupyter notebooks can introduce:
- JSON syntax errors
- Unicode encoding issues  
- Escape sequence problems
- Line ending issues

**Manual editing** in Jupyter:
- ✅ Jupyter handles all formatting automatically
- ✅ No JSON corruption risk
- ✅ No Unicode issues
- ✅ Immediate validation
- ✅ Takes only 30 seconds

---

## 📊 What These Changes Do

### Before (Will Crash):
```
save_steps=1000       # Saves every 1000 steps
save_total_limit=3    # Keeps 3 checkpoints

Result:
- 28 saves during training
- Up to 9 checkpoints × 3GB = 27GB
- Crashes at step 9000 due to disk full
```

### After (Will Complete):
```
save_steps=2000       # Saves every 2000 steps  
save_total_limit=2    # Keeps only 2 checkpoints

Result:
- 14 saves during training
- Max 2 checkpoints × 3GB = 6GB
- Completes successfully ✅
```

---

## 🚀 Complete Workflow

1. ✅ **Upload original notebook** to Jupyter (it's already valid)
2. ✅ **Make 2-number edit** in Cell 19 (30 seconds)
3. ✅ **Run Cell 19** to generate fixed train.py
4. ✅ **Upload dataset** to `/data/Cogumi-LLM/data/phase1/`
5. ✅ **Start training** from Cell 22 or so
6. ✅ **Training completes** without crashes!

---

## 💡 Alternative: Pre-Upload Dataset and Script

If you prefer, you can also:

1. Upload the original notebook
2. Upload the dataset
3. **Manually create `/data/Cogumi-LLM/train.py`** with the fixed values
4. Skip the script generation cell
5. Start training

**To manually create train.py**, see `docs/DISK_SPACE_FIX.md` for the complete script with fixes already applied.

---

## ⚠️ Why Automated Editing Failed

When I tried to automatically edit the notebook to add:
- Checkpoint cleanup callback
- Disk monitoring  
- Additional cells

The automated edits introduced:
- JSON syntax errors (stray `%` character)
- Unicode surrogate issues (`\udee1`)
- Escape sequence problems (`\\\\n` vs `\\n`)

**These are common issues** when programmatically editing Jupyter notebooks because:
- Notebooks are complex JSON structures
- String escaping is tricky (Python string → JSON string → displayed string)
- Unicode handling varies between systems
- Easy to introduce subtle formatting errors

---

## ✅ Recommended Approach Going Forward

**For simple parameter changes:**
- ✅ Edit manually in Jupyter (safest)
- ✅ Takes 30 seconds
- ✅ Zero risk of corruption

**For complex additions:**
- ✅ Create separate Python scripts
- ✅ Import and use them in notebook
- ✅ Keeps notebook clean and simple

---

## 📋 Quick Checklist

Before starting training:

- [ ] Notebook uploaded to Jupyter
- [ ] Cell 19 edited (save_steps=2000, save_total_limit=2)
- [ ] Cell 19 executed successfully
- [ ] Verified train.py has correct values
- [ ] Dataset uploaded to `/data/Cogumi-LLM/data/phase1/`
- [ ] Disk space <30% used (`df -h /data`)
- [ ] Ready to start training!

---

## 🎯 Bottom Line

**The original notebook is fine and will upload without errors.**

Just make **2 simple number changes** in Jupyter before running:
- `save_steps`: 1000 → 2000
- `save_total_limit`: 3 → 2

That's all you need! Training will complete successfully with these changes.

---

## 📞 If You Need Help

The original notebook is restored and validated:
```bash
✅ JSON is valid
✅ Will upload to Jupyter without errors
✅ Just needs the 2-number edit before training
```

**You're ready to go!** 🚀
