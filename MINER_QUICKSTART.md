# Miner Quick Start Checklist

## ✅ Pre-Flight Checklist

### 1. Prerequisites
- [ ] Python 3.10+ installed
- [ ] Docker installed
- [ ] Bittensor CLI (`btcli`) installed
- [ ] GitHub account created
- [ ] Bittensor wallet with TAO

### 2. Wallet Setup
```bash
btcli wallet new_coldkey --wallet.name default  # if needed
btcli wallet new_hotkey --wallet.name default --wallet.hotkey miner
```

### 3. Solution Repository
- [ ] Fork/copy `miner-solution-example/` structure
- [ ] Implement your ARC solver
- [ ] Push to GitHub (must be public)
- [ ] Note your repo URL, branch, and commit

### 4. Miner Configuration
Create `miner/.env`:
```ini
WALLET_NAME=default
WALLET_HOTKEY=miner
MINER_PORT=8091
MINER_REPO_URL=https://github.com/yourusername/your-repo
MINER_REPO_BRANCH=main
MINER_WEIGHT_CLASS=1xH200
MINER_USE_VLLM=true
VLLM_MODEL=unsloth/Meta-Llama-3.1-8B-Instruct
```

### 5. Register on Chain
```bash
# Register miner
btcli subnet register --netuid 5 --wallet.name default --wallet.hotkey miner

# Set IP/port (replace with your public IP)
python tools/post_ip_chain.py \
  --wallet-name default \
  --hotkey miner \
  --ip $(curl -s ifconfig.me) \
  --port 8091
```

### 6. Start Miner
```bash
docker build -t hone-miner miner/
docker run -d --name miner \
  -p 8091:8091 \
  -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
  --env-file miner/.env \
  hone-miner
```

### 7. Verify
```bash
# Check miner is running
docker ps | grep miner

# Test /info endpoint
curl http://localhost:8091/info

# Check logs
docker logs -f miner
```

---

## 🔍 Common Issues

| Issue | Solution |
|-------|----------|
| Miner not discovered | Check IP/port on-chain, firewall rules |
| `/info` not responding | Check docker logs, port binding |
| Sandbox jobs fail | Verify repo is public, check Dockerfile |
| Low scores | Review solver logic, test locally |

---

## 📚 Full Documentation

See `MINER_WORKFLOW.md` for detailed explanations.

