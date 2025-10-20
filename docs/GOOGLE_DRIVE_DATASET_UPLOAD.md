# Google Drive Dataset Upload Guide - Fastest Method! ⚡

**Purpose:** Upload dataset to Google Colab from Google Drive  
**Time:** ~2-3 minutes (vs 9-10 min local upload)  
**File:** `public_500k_filtered.jsonl.gz` (264 MB)

---

## 🎯 Why Use Google Drive?

| Method | Time | Speed |
|--------|------|-------|
| ✅ **Google Drive** | **2-3 min** | **FASTEST!** (server-to-server) |
| Local Upload (compressed) | 9-10 min | Fast (upload required) |
| Local Upload (uncompressed) | 30-35 min | Slow (large file) |

**Benefits:**
- ✅ 3-5x faster than local upload
- ✅ No network bandwidth usage from your computer
- ✅ More reliable for large files
- ✅ Can access from any Colab session

---

## 📋 Method 1: Using Google Drive File ID (Recommended)

### Step 1: Upload File to Google Drive

1. Open [Google Drive](https://drive.google.com)
2. Upload `public_500k_filtered.jsonl.gz` to any folder
3. Wait for upload to complete

### Step 2: Get File ID

**Option A: From Share Link**
1. Right-click the file → **Get link**
2. Click **Copy link**
3. Extract FILE_ID from URL:
   ```
   URL: https://drive.google.com/file/d/1ABC123XYZ456/view?usp=sharing
   FILE_ID: 1ABC123XYZ456
   ```

**Option B: From File URL**
1. Open the file in Google Drive
2. Look at the browser URL bar
3. Copy the ID between `/d/` and `/view`:
   ```
   https://drive.google.com/file/d/1ABC123XYZ456/view
                                    ↑ This is the FILE_ID
   ```

### Step 3: Update Colab Notebook

In **Section 3b, Option 1** cell, find this line:
```python
FILE_ID = "YOUR_FILE_ID_HERE"
```

Replace with your actual FILE_ID:
```python
FILE_ID = "1ABC123XYZ456"
```

### Step 4: Uncomment Download Lines

Find these commented lines:
```python
# !gdown --id {FILE_ID} -O data/phase1/public_500k_filtered.jsonl.gz
# !gunzip data/phase1/public_500k_filtered.jsonl.gz
# print("\n✅ Download and decompression complete!")
```

Uncomment them (remove the `#`):
```python
!gdown --id {FILE_ID} -O data/phase1/public_500k_filtered.jsonl.gz
!gunzip data/phase1/public_500k_filtered.jsonl.gz
print("\n✅ Download and decompression complete!")
```

### Step 5: Run the Cell

Click **Run cell** and wait ~2-3 minutes for download + decompression.

---

## 📋 Method 2: Using Drive Path (Alternative)

### Step 1: Mount Google Drive

The cell already includes:
```python
from google.colab import drive
drive.mount('/content/drive')
```

This will prompt you to authorize access.

### Step 2: Find Your File Path

In Colab's file browser (left sidebar):
1. Click on **📁 drive** → **MyDrive**
2. Navigate to where you uploaded the file
3. Right-click the file → **Copy path**

Example path:
```
/content/drive/MyDrive/Datasets/public_500k_filtered.jsonl.gz
```

### Step 3: Update the Cell

Find these commented lines:
```python
# !cp /content/drive/MyDrive/YOUR_PATH/public_500k_filtered.jsonl.gz data/phase1/
# !gunzip data/phase1/public_500k_filtered.jsonl.gz
# print("\n✅ Copy and decompression complete!")
```

Update with your actual path and uncomment:
```python
!cp /content/drive/MyDrive/Datasets/public_500k_filtered.jsonl.gz data/phase1/
!gunzip data/phase1/public_500k_filtered.jsonl.gz
print("\n✅ Copy and decompression complete!")
```

### Step 4: Run the Cell

Click **Run cell** and wait ~2-3 minutes.

---

## 🔍 Troubleshooting

### Error: "gdown: command not found"

**Solution:** Install gdown first:
```python
!pip install gdown
```

Then run the download command again.

---

### Error: "Access denied" or "Cannot access file"

**Solution:** Make sure file is shared properly:
1. Right-click file in Google Drive
2. Click **Share**
3. Under "General access" select **Anyone with the link**
4. Click **Copy link** and extract FILE_ID
5. Update FILE_ID in the cell
6. Run cell again

---

### Error: "File not found in Drive path"

**Solution:** Double-check the path:
1. In Colab file browser, navigate to the file
2. Right-click → **Copy path**
3. Paste exact path in the command
4. Make sure Drive is mounted first

---

### Download is slow

**Possible causes:**
- Large file (gdown limit): Try Method 2 (Drive path) instead
- Drive quota exceeded: Check your Google Drive storage
- Network issues: Wait and retry

---

## 📊 Verification

After download completes, verify the dataset:

```python
# Check file exists and size
!ls -lh data/phase1/public_500k_filtered.jsonl

# Count lines (should be 640,637)
!wc -l data/phase1/public_500k_filtered.jsonl

# Check first example
import json
with open('data/phase1/public_500k_filtered.jsonl', 'r') as f:
    example = json.loads(f.readline())
    print(f"Keys: {list(example.keys())}")
    print(f"Instruction: {example['instruction'][:100]}...")
    print(f"Response: {example['response'][:100]}...")
```

**Expected output:**
```
-rw-r--r-- 1 root root 870M Oct 20 12:34 data/phase1/public_500k_filtered.jsonl
640637 data/phase1/public_500k_filtered.jsonl
Keys: ['instruction', 'response', 'source', 'language']
Instruction: Write a Python function to calculate the factorial of a number...
Response: Here's a Python function that calculates the factorial of a number...
```

---

## ⚡ Quick Reference

### Method 1: Using FILE_ID (gdown)
```python
# 1. Get FILE_ID from Google Drive link
FILE_ID = "1ABC123XYZ456"

# 2. Download and decompress
!gdown --id {FILE_ID} -O data/phase1/public_500k_filtered.jsonl.gz
!gunzip data/phase1/public_500k_filtered.jsonl.gz
```

### Method 2: Using Drive Path (cp)
```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Copy and decompress
!cp /content/drive/MyDrive/YOUR_PATH/public_500k_filtered.jsonl.gz data/phase1/
!gunzip data/phase1/public_500k_filtered.jsonl.gz
```

---

## 🎯 Recommended Workflow

1. **Before Colab Session:** Upload `public_500k_filtered.jsonl.gz` to Google Drive
2. **During Setup:** Use Method 1 (FILE_ID) for fastest download
3. **Verify:** Run verification cell to confirm dataset integrity
4. **Proceed:** Continue with Section 4 (HuggingFace Authentication)

---

## 📚 Related Documentation

- [Phase1A Training Colab Notebook](../notebooks/Phase1A_Training_Colab.ipynb)
- [Dependency Compatibility Matrix](./DEPENDENCY_COMPATIBILITY_MATRIX.md)
- [Colab Setup Guide](./COLAB_PRO_PLUS_GUIDE.md)

---

**Last Updated:** 2024-10-20  
**Status:** ✅ Ready to use
