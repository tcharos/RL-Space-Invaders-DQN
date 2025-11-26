# AIDL_B02-Advanced-Topics-in-Deep-Learning

Space Invaders

Implementation Supports
- ✅ Standard DQN
- ✅ Double DQN  
- ✅ Dueling DQN
- ✅ Prioritized Experience Replay (PER)
- ✅ Google Drive integration (optional)
- ✅ Automatic checkpointing
- ✅ Memory/GPU monitoring
- ✅ Consolidated plotting


## 🚀 Key Features Implemented

### 1. Configuration System

CONFIG = {
    'DQN_TYPE': 'DoubleDQN',  # Easy to switch variants
    'USE_PER': False,
    'SEED': 42,
    'CHECKPOINT_EVERY': 100,
    'USE_GDRIVE': True,
    # ... all hyperparameters
}

### 2. Automatic Checkpointing

**Regular checkpoints:**
- Every 100 episodes (configurable)
- Filename: `DoubleDQN_ep500_20241126_143052.pth`

**Best model checkpoints:**
- Saved when average improves
- Filename: `DoubleDQN_best.pth`
- Always keeps best performing model

**Each checkpoint contains:**
- Model weights (policy + target)
- Optimizer state
- Full configuration
- Episode rewards history
- Current average score
- Timestamp

### 3. Memory Monitoring
Every 10 episodes:
```
Episode 100	Score: 125.0	Avg: 98.45	Eps: 0.904	Steps: 15432
[Ep 100] RAM: 45.2% | GPU: 1.23GB | Buffer: 10000/10000
Checkpoint saved: DoubleDQN_ep100_20241126_143052.pth
```

## 🔬 DQN Variants Explained

### Standard DQN
**What it does:**
- Experience replay for stable learning
- Target network to reduce correlation
- Epsilon-greedy exploration

**When to use:**
- Baseline comparison
- Understanding fundamentals

### Double DQN
**What's different:**
- Uses policy net to SELECT actions
- Uses target net to EVALUATE actions
- Reduces Q-value overestimation

**Why it's better:**
- More stable training
- Better final performance
- Minimal computational overhead

**Key code change:**
```python
# Standard DQN
next_q = target_net(next_states).max(1)[0]

# Double DQN
next_actions = policy_net(next_states).max(1)[1]  # Select
next_q = target_net(next_states).gather(1, next_actions)  # Evaluate
```

### Dueling DQN
**What's different:**
- Splits network into two streams:
  - Value stream: V(s) - "How good is this state?"
  - Advantage stream: A(s,a) - "How much better is action a?"
- Combines: Q(s,a) = V(s) + (A(s,a) - mean(A))

### Prioritized Experience Replay (PER)
**What's different:**
- Samples transitions based on TD error (priority)
- High TD error = more to learn = sampled more often
- Uses importance sampling weights to correct bias

**Why it's better:**
- More efficient learning
- Focuses on important experiences
- Can achieve same performance with less data


## 📊 How to Use

### Quick Start - Single Variant

```python
# 1. Choose variant
CONFIG = BASE_CONFIG.copy()
CONFIG['DQN_TYPE'] = 'DoubleDQN'
CONFIG['USE_PER'] = False

# 2. Initialize
policy_net = DQN((4, 84, 84), 6).to(device)
target_net = DQN((4, 84, 84), 6).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=CONFIG['LEARNING_RATE'])
replay_buffer = ReplayBuffer(CONFIG['BUFFER_SIZE'])

# 3. Train
rewards = train_dqn(CONFIG, policy_net, target_net, optimizer, replay_buffer, device)

# 4. Store
all_results['DoubleDQN'] = rewards
```

### Running All 4 Variants

The notebook has pre-configured cells for all variants. Just run in sequence:

1. **Cell: "Initialize and Train - Standard DQN"**
   - Uses seed=7
   - Baseline performance

2. **Cell: "Initialize and Train - Double DQN"**
   - Uses seed=42
   - Improved stability

3. **Cell: "Initialize and Train - Dueling DQN"**
   - Uses seed=123
   - Different architecture

4. **Cell: "Initialize and Train - DQN with PER"**
   - Uses seed=456
   - Priority sampling

5. **Cell: "Plot All Results"**
   - Consolidated comparison plot
   - Individual plots for each variant

### Customizing Hyperparameters

```python
# Example: Aggressive exploration
CONFIG_CUSTOM = BASE_CONFIG.copy()
CONFIG_CUSTOM['DQN_TYPE'] = 'DoubleDQN'
CONFIG_CUSTOM['EPSILON_DECAY'] = 20000  # Slower decay
CONFIG_CUSTOM['LEARNING_RATE'] = 0.0001  # Lower LR
CONFIG_CUSTOM['BUFFER_SIZE'] = 50000  # Larger buffer

# Train with custom config
rewards_custom = train_dqn(CONFIG_CUSTOM, ...)
```

## 🔧 Maintenance & Continuation

### Resuming Training

```python
# Load checkpoint
checkpoint = torch.load('checkpoints/DoubleDQN_best.pth')

# Restore everything
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
target_net.load_state_dict(checkpoint['target_net_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
episode_rewards = checkpoint['episode_rewards']
config = checkpoint['config']

# Continue training from this point
# (you'd need to modify train_dqn to accept start_episode)
```

### Adding New Variants

To add new DQN variants (e.g., Rainbow DQN):

1. **Create new network class** (if needed)
2. **Add optimization function** (if needed)
3. **Create config:**
   ```python
   CONFIG_RAINBOW = BASE_CONFIG.copy()
   CONFIG_RAINBOW['DQN_TYPE'] = 'RainbowDQN'
   ```
4. **Train and store results:**
   ```python
   all_results['RainbowDQN'] = train_dqn(...)
   ```

## 📝 Quick Commands Reference

```python
# Enable Google Drive
USE_GDRIVE = True

# Choose DQN variant
CONFIG['DQN_TYPE'] = 'DoubleDQN'  # or 'DQN', 'DuelingDQN'

# Enable PER
CONFIG['USE_PER'] = True

# Train
rewards = train_dqn(CONFIG, policy_net, target_net, optimizer, replay_buffer, device)

# Store results
all_results['MyVariant'] = rewards

# Plot consolidated
plot_consolidated_results(all_results)

# Save everything
with open('all_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)
```
