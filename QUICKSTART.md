# Quick Start Guide

Get started with the Post-Silicon Semiconductor GPU Analysis in just 5 minutes!

## 🚀 Fastest Way to Run (Google Colab with GPU)

### 1. Click the Badge

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajendarmuddasani/post-silicon-gpu-analysis/blob/main/semiconductor_analysis_cpu_vs_gpu.ipynb)

### 2. Enable GPU

In Colab: **Runtime → Change runtime type → GPU → Save**

### 3. Clone Repository

Add this as the first cell and run it:

```python
!git clone https://github.com/rajendarmuddasani/post-silicon-gpu-analysis.git
%cd post-silicon-gpu-analysis
```

### 4. Run All Cells

**Runtime → Run all** (or press `Ctrl+F9`)

### 5. See Results

Scroll to the bottom to see:
- CPU vs GPU execution times
- Speedup metrics (how much faster GPU is)
- Performance comparison charts
- Time saved calculations

## 💻 Run on Your Laptop First (CPU Only)

Want to compare your laptop's CPU performance with Colab's GPU?

```bash
# Clone the repo
git clone https://github.com/rajendarmuddasani/post-silicon-gpu-analysis.git
cd post-silicon-gpu-analysis

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# Generate data (if needed)
python generate_data.py

# Start notebook
jupyter notebook semiconductor_analysis_cpu_vs_gpu.ipynb
```

Run all cells and **note the execution times**. Then upload to Colab with GPU to see the speedup!

## 📊 What You'll See

### CPU Performance (Typical Laptop)
- Matrix Operations: ~15-25 seconds
- Statistical Analysis: ~3-5 seconds
- PCA: ~8-12 seconds
- **Total: ~50-70 seconds**

### GPU Performance (Colab T4)
- Matrix Operations: ~2-4 seconds ⚡
- Statistical Analysis: ~0.5-1 seconds ⚡
- PCA: ~1-2 seconds ⚡
- **Total: ~25-35 seconds**

### Speedup
**2-3x faster overall** with GPU! 🚀

## 🎯 Key Features

✅ **Automatic hardware detection** - Works on both CPU and GPU  
✅ **Large dataset** - 50,000 chips × 100 parameters (~90 MB)  
✅ **Real-world tasks** - Matrix ops, statistics, PCA, ML  
✅ **Visual comparisons** - Charts showing CPU vs GPU performance  
✅ **Detailed explanations** - Markdown cells explain every step  

## 📚 Need More Help?

- **Detailed Instructions**: See [COLAB_INSTRUCTIONS.md](COLAB_INSTRUCTIONS.md)
- **Full Documentation**: See [README.md](README.md)
- **Issues**: Open an issue on GitHub

## 🎓 What You'll Learn

1. How GPU acceleration works
2. When to use GPU vs CPU
3. How to measure performance
4. Semiconductor data analysis techniques
5. Google Colab GPU usage

**Happy Computing! 🎉**
