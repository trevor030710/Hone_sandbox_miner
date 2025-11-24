# Miner Setup Workflow Guide

This guide walks you through the complete process of becoming a miner on the Hone subnet.

## Overview

The Hone subnet uses a **sandbox-based architecture**:
- Miners run a FastAPI server that exposes an `/info` endpoint
- The `/info` endpoint tells validators where your solution code lives (GitHub repo)
- Validators submit jobs to a sandbox runner that clones your repo and executes your solution
- Your solution is evaluated on ARC-AGI-2 problems

---

## Step 1: Prerequisites

### Required Software
- **Python 3.10+**
- **Docker & Docker Compose**
- **Bittensor CLI** (`btcli`) installed
- **Git** for version control

### Required Accounts
- A **GitHub account** (to host your solution repository)
- A **Bittensor wallet** with some TAO (for registration)

---

## Step 2: Create Your Bittensor Wallet

If you don't have a wallet yet:

```bash
# Create coldkey (if needed)
btcli wallet new_coldkey --wallet.name default

# Create miner hotkey
btcli wallet new_hotkey --wallet.name default --wallet.hotkey miner
```

Your wallet files will be stored in `~/.bittensor/wallets/default/`.

---

## Step 3: Create Your Solution Repository

### 3.1 Fork/Copy the Example Solution

The `miner-solution-example/` directory shows the expected structure:

```
your-miner-repo/
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── arc_main.py            # Main entry point (CLI wrapper)
├── arc_prep_phase.py      # Prep phase (download models, etc.)
├── arc_inference_phase.py # Inference phase (solve problems)
├── arc_solver_llm.py      # Your solver implementation
└── arc_utils.py           # Helper utilities
```

### 3.2 Key Files Explained

**`arc_main.py`**: Entry point that supports:
```bash
python arc_main.py --phase prep --input ... --output ...
python arc_main.py --phase inference --input ... --output ...
```

**`arc_prep_phase.py`**: 
- Runs in a **prep container** with internet access
- Download models, weights, datasets, etc.
- Save assets to `/app/models` or similar
- These assets are shared with the inference container

**`arc_inference_phase.py`**:
- Runs in an **inference container** with **NO internet access**
- Loads assets from prep phase
- Solves ARC-AGI-2 problems
- Must output `results.json` with this structure:
```json
{
  "phase": "inference",
  "status": "success",
  "predictions": [
    {
      "problem_index": 0,
      "task_hash": "...",
      "predicted_output": [[0, 1, 2], [1, 2, 0], ...]
    }
  ]
}
```

**`arc_solver_llm.py`**: 
- Replace this with your own solver
- Can use LLMs (vLLM), heuristics, program synthesis, etc.
- Must implement a `solve()` method that takes `train_examples` and `test_input`

### 3.3 Push to GitHub

```bash
cd your-miner-repo
git init
git add .
git commit -m "Initial miner solution"
git remote add origin https://github.com/yourusername/your-miner-repo.git
git push -u origin main
```

**Important**: Make sure your repo is **public** or the sandbox runner won't be able to clone it.

---

## Step 4: Configure Your Miner Server

### 4.1 Set Environment Variables

Create `miner/.env`:

```ini
# ---- Wallet ----
WALLET_NAME=default
WALLET_HOTKEY=miner

# ---- Server ----
HOST=0.0.0.0
MINER_PORT=8091
LOG_LEVEL=INFO

# ---- Solution Repository ----
MINER_REPO_URL=https://github.com/yourusername/your-miner-repo
MINER_REPO_BRANCH=main
MINER_REPO_COMMIT=                    # Optional: specific commit hash
MINER_REPO_PATH=                      # Optional: subdirectory within repo

# ---- Resource Requirements ----
MINER_WEIGHT_CLASS=1xH200             # Options: 1xH200, 2xH200, 4xH200, 8xH200

# ---- vLLM Configuration (if using) ----
MINER_USE_VLLM=true
VLLM_MODEL=unsloth/Meta-Llama-3.1-8B-Instruct
VLLM_DTYPE=half
VLLM_GPU_MEMORY_UTIL=0.8
VLLM_MAX_MODEL_LEN=12000

# ---- Custom Environment Variables (optional) ----
MINER_CUSTOM_ENV_VARS=OPENAI_API_KEY=sk-...,ANOTHER_VAR=value

# ---- Version ----
MINER_VERSION=1.0.0

# ---- Testing (production: false) ----
SKIP_EPISTULA_VERIFY=false
```

### 4.2 Key Configuration Fields

- **`MINER_REPO_URL`**: Your GitHub repository URL
- **`MINER_REPO_BRANCH`**: Branch to use (default: `main`)
- **`MINER_REPO_COMMIT`**: Optional specific commit (leave empty for latest)
- **`MINER_REPO_PATH`**: Optional subdirectory if your solution is in a subfolder
- **`MINER_WEIGHT_CLASS`**: GPU resources needed (`1xH200` = 1 GPU, `2xH200` = 2 GPUs, etc.)
- **`MINER_USE_VLLM`**: Whether you need vLLM for model serving
- **`VLLM_MODEL`**: Model identifier if using vLLM

---

## Step 5: Register on Bittensor Chain

### 5.1 Register Your Miner

```bash
btcli subnet register \
  --netuid 5 \
  --wallet.name default \
  --wallet.hotkey miner
```

This registers your miner identity on subnet 5 (Hone subnet).

### 5.2 Set Your Public IP and Port

You need to tell the chain where your miner server is running:

```bash
python tools/post_ip_chain.py \
  --wallet-name default \
  --hotkey miner \
  --ip YOUR_PUBLIC_IP \
  --port 8091
```

**Important**: 
- Use your **public IP address** (not localhost)
- Ensure your firewall allows inbound traffic on port 8091
- If behind NAT, configure port forwarding

To find your public IP:
```bash
curl ifconfig.me
```

---

## Step 6: Start Your Miner Server

### 6.1 Build and Run with Docker

```bash
# Build the miner image
docker build -t hone-miner miner/

# Run the miner
docker run -d --name miner \
  -p 8091:8091 \
  -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
  --env-file miner/.env \
  hone-miner

# Check logs
docker logs -f miner
```

### 6.2 Verify It's Running

Test the `/info` endpoint:

```bash
curl http://localhost:8091/info
```

You should see JSON with your repository URL, weight class, etc.

---

## Step 7: How It Works

### 7.1 Discovery Flow

1. **Validators discover miners** from the Bittensor chain
2. **Validators fetch `/info`** from each miner's IP:port
3. **Validators submit jobs** to the sandbox runner with:
   - Your repo URL, branch, commit
   - Your weight class (GPU requirements)
   - vLLM config (if needed)

### 7.2 Sandbox Execution Flow

1. **Sandbox runner clones** your repository
2. **Prep phase runs** (with internet):
   - Downloads models, datasets, etc.
   - Saves assets to shared storage
3. **Inference phase runs** (no internet):
   - Loads assets from prep phase
   - Solves ARC-AGI-2 problems
   - Outputs `results.json` with predictions
4. **Results are evaluated**:
   - Exact match rate
   - Partial correctness
   - Grid similarity
   - Efficiency score

### 7.3 Scoring

Your miner is scored on:
- **Exact match** (≈40% weight): Perfect solutions
- **Partial correctness** (≈30%): Partial credit for correct parts
- **Grid similarity** (≈20%): Pixel-level similarity
- **Efficiency** (≈10%): Response time and resource usage

---

## Step 8: Testing Locally

Before going live, test your setup:

```bash
# Run the local test script
chmod +x test_local.sh
./test_local.sh
```

This will:
- Start a local Postgres database
- Launch 3 mock miners
- Run a validator with mock chain
- Test the full cycle

---

## Step 9: Monitoring and Updates

### 9.1 Check Your Miner Status

Your miner should appear in validator logs. You can also check:
- Validator database (if you have access)
- Bittensor explorer for your subnet

### 9.2 Update Your Solution

1. **Update your code** in your GitHub repo
2. **Update `MINER_REPO_COMMIT`** in `.env` (or leave empty for latest)
3. **Restart your miner**:
   ```bash
   docker restart miner
   ```

The next evaluation cycle will use your updated code.

---

## Troubleshooting

### Miner Not Discovered

- ✅ Check you registered on the correct `netuid` (5)
- ✅ Verify your IP/port are set correctly on-chain
- ✅ Ensure firewall allows inbound traffic on your port
- ✅ Check miner logs: `docker logs miner`

### `/info` Endpoint Not Responding

- ✅ Verify miner is running: `docker ps`
- ✅ Check port is exposed: `curl http://localhost:8091/info`
- ✅ Review logs for errors: `docker logs miner`

### Sandbox Jobs Failing

- ✅ Ensure your repo is **public**
- ✅ Verify your `Dockerfile` builds successfully
- ✅ Check that `arc_main.py` supports `--phase prep` and `--phase inference`
- ✅ Verify `results.json` format matches expected structure

### Low Scores

- ✅ Review your solver implementation
- ✅ Test locally with `test_local.sh`
- ✅ Check sandbox runner logs for execution errors
- ✅ Verify models/assets are loading correctly in prep phase

---

## Example Solution Structure

Here's a minimal working example:

**`Dockerfile`**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "arc_main.py"]
```

**`arc_main.py`**:
```python
import argparse
from pathlib import Path
from arc_prep_phase import run_prep_phase
from arc_inference_phase import run_inference_phase

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prep", "inference"], required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    if args.phase == "prep":
        run_prep_phase()
    else:
        run_inference_phase(Path(args.input), Path(args.output))
```

**`arc_inference_phase.py`**:
```python
from pathlib import Path
import json

def run_inference_phase(input_dir: Path, output_dir: Path):
    # Load problems from input_dir
    with open(input_dir / "tasks.json") as f:
        tasks = json.load(f)
    
    predictions = []
    for i, task in enumerate(tasks["tasks"]):
        # Your solver logic here
        predicted_output = your_solver.solve(
            train_examples=task["train_examples"],
            test_input=task["test_input"]
        )
        
        predictions.append({
            "problem_index": i,
            "task_hash": task.get("task_hash"),
            "predicted_output": predicted_output
        })
    
    # Save results
    results = {
        "phase": "inference",
        "status": "success",
        "predictions": predictions
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f)
```

---

## Next Steps

1. ✅ Create your solution repository
2. ✅ Implement your ARC solver
3. ✅ Configure your miner server
4. ✅ Register on-chain
5. ✅ Start your miner
6. ✅ Monitor performance
7. ✅ Iterate and improve!

Good luck! 🚀

