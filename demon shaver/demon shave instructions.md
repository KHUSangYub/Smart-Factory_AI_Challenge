# 👺 Demon Shaving Optimization Toolkit

This toolkit is designed to help you hit the **1-second inference target** by identifying and removing "dead" neurons from your model. These are neurons that consistently output zero and waste CPU cycles.

## 🛠️ Components

### 1. [demon_shave_target_finder.py](./demon_shave_target_finder.py)
**The Scout** 🕵️‍♂️
*   **Purpose:** Scans your model to find neurons that never "wake up" across your dataset.
*   **How it works:** It uses Forward Hooks to monitor activations (ReLU/Hardswish).
*   **Output:** Creates `dead_neurons.json` containing the indices of all dead neurons.

### 2. [demon_shave_tester.py](./demon_shave_tester.py)
**The Simulation** 🧪
*   **Purpose:** Performs "Soft Shaving" by setting dead neurons to zero.
*   **Why use it:** Verify that removing these neurons doesn't hurt your accuracy.
*   **Note:** This will **NOT** improve your speed yet, as the CPU still calculates `x * 0 = 0`.

### 3. [demon_shaver.py](./demon_shaver.py)
**The Surgeon** 🔪
*   **Purpose:** Performs **Physical Shaving** (Pruning).
*   **Speedup:** This actually deletes the neurons from the model's architecture, reducing the number of calculations. **This is what will help you hit the 1s limit.**
*   **Requirement:** Requires the `torch-pruning` library.
    ```bash
    pip install torch-pruning
    ```

---

## 🚀 Step-by-Step Workflow

### Step 1: Training
Complete your training phase until you have a high-accuracy `.pth` file.

### Step 2: Diagnosis
Run the target finder to see if your model has "Demons" (dead weight).
```bash
python demon_shave_target_finder.py
```

### Step 3: Safety Check
Run the tester to see if your accuracy holds up with the neurons masked.
```bash
python demon_shave_tester.py
```

### Step 4: Final Shave
Run the shaver to physically shrink the model and gain a massive speed boost.
```bash
python demon_shaver.py
```

---

## 💡 Pro Tip for the Group
Mention this to your teammates as a **"Structured Pruning"** or **"ReLU Sparsity Analysis"** strategy. It’s an advanced optimization technique used in industry-level Edge AI deployment!
