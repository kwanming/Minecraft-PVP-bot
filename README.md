# Minecraft-PVP-bot
Minecraft Deep Reinforcement Learning PVP AI bot

# 🤖 Minecraft PvP Bot - Deep Reinforcement Learning

Train an AI bot to play Minecraft PvP using Deep Reinforcement Learning (PPO algorithm). The bot learns by watching the screen and controlling the game directly - no mods or server plugins required!

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

## ✨ Features

- 🎮 **Pure Python** - No Node.js or bridge servers needed
- 🖥️ **Direct Control** - Controls Minecraft via keyboard/mouse simulation
- 👁️ **Vision-Based Learning** - Learns from screen capture using CNN
- 🧠 **Deep RL** - Uses Proximal Policy Optimization (PPO)
- ⚔️ **Full PvP Actions** - Movement, mouse aiming, attacking, strafing
- 📊 **TensorBoard Integration** - Monitor training progress in real-time
- 🔧 **Easy Setup** - Get started in under 5 minutes

## 📋 Table of Contents

- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)

## 🔬 How It Works

```
Screen Capture → CNN → PPO Agent → Actions → Game Control
     ↓                                            ↓
     └────────────── Reward Signal ←─────────────┘
```

1. **Captures** your Minecraft screen in real-time (84x84 RGB)
2. **Processes** through Convolutional Neural Network
3. **Decides** best action using trained policy
4. **Executes** keyboard/mouse commands
5. **Learns** from rewards (damage dealt, survival, kills)

### Action Space (16 Actions)

| Action | Description | Action | Description |
|--------|-------------|--------|-------------|
| 0 | No-op | 8 | Crouch |
| 1 | Forward | 9 | Look Left |
| 2 | Backward | 10 | Look Right |
| 3 | Strafe Left | 11 | Look Up |
| 4 | Strafe Right | 12 | Look Down |
| 5 | Attack | 13 | Forward + Attack |
| 6 | Jump | 14 | Strafe Left + Attack |
| 7 | Sprint | 15 | Strafe Right + Attack |

## 💻 Installation

### Prerequisites

- **Minecraft Java Edition**
- **Windows, macOS, or Linux**
- **4GB+ RAM** (8GB+ recommended for training)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/minecraft-pvp-bot.git
cd minecraft-pvp-bot
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- `torch` - Deep learning framework
- `stable-baselines3` - RL algorithms (PPO)
- `gymnasium` - RL environment interface
- `opencv-python` - Image processing
- `mss` - Fast screen capture
- `pydirectinput` - Keyboard/mouse control
- `tensorboard` - Training visualization

### Step 3: Install PyTorch with GPU (Optional but Recommended)

For faster training with NVIDIA GPU:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 🚀 Quick Start

### 1. Test Controls (Recommended First Step)

- Choose option **2** (Interactive test)
- Open Minecraft first
- Watch as bot tests movements and mouse control

### 2. Find Minecraft Window

```bash
python find_minecraft.py
```

- Choose option **2** (Interactive mode)
- Use **W/A/S/D** to position capture box over Minecraft
- Use **+/-** to resize
- Press **Q** when perfectly aligned
- Copy the coordinates printed

Example output:
```python
capture_region = {'top': 300, 'left': 533, 'width': 854, 'height': 480}
```

### 3. Configure Capture Region

Edit `main.py` at the bottom:

```python
# Replace with your coordinates from step 2
capture_region = {'top': 300, 'left': 533, 'width': 854, 'height': 480}

train_pvp_bot(capture_region=capture_region, total_timesteps=100_000)
```

### 4. Start Minecraft

1. Launch **Minecraft Java Edition**
2. Use **Windowed Mode** (not fullscreen) - This is critical!
3. Join a world/server with **PvP enabled**
4. Position window where you set the capture region

### 5. Start Training

```bash
python main.py
```

**Important:** Don't touch keyboard/mouse during training!

## ⚙️ Configuration

### Basic Configuration

```python
# Capture region (from find_minecraft.py)
capture_region = {'top': 300, 'left': 533, 'width': 854, 'height': 480}

# Training parameters
total_timesteps = 100_000  # More = better but slower

# PPO hyperparameters (in pure_python_bot.py)
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
```

### Advanced Configuration

```python
# Adjust in MinecraftPvPEnv class
self.max_steps = 1000  # Max steps per episode
time.sleep(0.05)  # Action execution delay (lower = faster but less stable)

# Adjust in _calculate_reward()
# Customize reward function for desired behavior
```

## 🎓 Training

### Training Process

```bash
python main.py
```

**What happens:**
- Bot starts exploring randomly
- Gradually learns patterns
- Improves over thousands of episodes
- Auto-saves model periodically

### Expected Timeline

| Steps | Behavior |
|-------|----------|
| 0-10k | Random exploration, learning basic controls |
| 10k-50k | Purposeful movement, occasional attacks |
| 50k-100k | Consistent combat, basic strategy |
| 100k+ | Advanced tactics, good decision-making |

### Monitor Training

```bash
tensorboard --logdir=./pvp_bot_tensorboard/
```

**Key metrics to watch:**
- **Reward**: Should trend upward
- **Episode Length**: Longer = surviving better
- **Loss**: Should decrease

### Training Tips

1. **Start Simple**: Train in peaceful mode or against single mob
2. **Curriculum Learning**: Gradually increase difficulty
3. **Longer Training**: 100k+ steps for good results
4. **Use GPU**: 3-5x faster training
5. **Adjust Rewards**: Tune reward function for desired behavior

## 🎯 Evaluation

Test your trained bot:

```bash
python main.py eval
```

Or programmatically:

```python
from pure_python_bot import evaluate_bot

evaluate_bot(
    model_path="minecraft_pvp_bot_pure_python",
    capture_region={'top': 300, 'left': 533, 'width': 854, 'height': 480},
    num_episodes=10
)
```

## 🐛 Troubleshooting

### Bot doesn't move in Minecraft

**Cause:** Input simulation not working

**Solutions:**
- Run Python as **Administrator** (Windows)
- Make sure Minecraft is in **Windowed mode**
- Check antivirus isn't blocking pydirectinput

### Screen capture shows wrong window

**Cause:** Incorrect capture region

**Solutions:**
- Re-run `find_minecraft.py`
- Make sure Minecraft window is visible (not minimized)
- Use windowed mode, not fullscreen
- Check if you have multiple monitors

### Bot just dies immediately

**Cause:** Not enough training or wrong environment

**Solutions:**
- Train in **peaceful mode** initially
- Use flat world with barriers
- Train longer (100k+ steps)
- Reduce death penalty in reward function

### Mouse not moving

**Cause:** pydirectinput not working properly

**Solutions:**
- Update: `pip install --upgrade pydirectinput`
- Run Python as Administrator
- Disable mouse acceleration in OS

### Low FPS / Laggy training

**Cause:** CPU bottleneck or slow capture

**Solutions:**
- Use GPU: Set `device='cuda'` in PPO
- Reduce capture region size
- Close other applications
- Lower Minecraft graphics settings
- Increase `time.sleep()` duration

## 🔧 Advanced Usage

### Custom Actions

Add new actions in `MinecraftController.execute_action()`:

```python
elif action == 16:  # Use item (right click)
    pydirectinput.rightClick()
elif action == 17:  # Switch to slot 1
    pydirectinput.press('1')
```

Don't forget to update action space:
```python
self.action_space = spaces.Discrete(18)  # Increase count
```

### Custom Reward Function

Edit `_calculate_reward()` in `MinecraftPvPEnv`:

```python
def _calculate_reward(self, screen):
    reward = 0.0
    
    # High reward for damage dealt
    reward += damage_dealt * 50
    
    # Bonus for maintaining high health
    reward += current_health * 2
    
    # Penalty for low health (encourages caution)
    if current_health < 0.3:
        reward -= 10
    
    return reward
```

### Multi-Agent Training

Train multiple bots against each other:

```python
# Create multiple environments
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(rank):
    def _init():
        return MinecraftPvPEnv(capture_region=regions[rank])
    return _init

envs = SubprocVecEnv([make_env(i) for i in range(4)])
model = PPO("CnnPolicy", envs, ...)
```

### Save/Load Models

```python
# Save during training (automatic)
model.save("my_bot_checkpoint")

# Load and continue training
model = PPO.load("my_bot_checkpoint")
model.learn(total_timesteps=50_000)

# Load for evaluation only
model = PPO.load("my_bot_checkpoint")
```

### Hyperparameter Tuning

```python
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=1e-4,      # Lower = more stable, slower
    n_steps=4096,            # Higher = more stable, needs more RAM
    batch_size=128,          # Higher = smoother updates
    n_epochs=20,             # More epochs = better learning per batch
    gamma=0.99,              # Future reward discount
    gae_lambda=0.95,         # Advantage estimation
    clip_range=0.2,          # PPO clipping
    ent_coef=0.01,           # Exploration bonus
    verbose=1
)
```

## 📊 Project Structure

```
minecraft-pvp-bot/
├── main.py          # Main training script
├── find_minecraft.py    # Window position finder
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── models/                     # Saved models
│   └── minecraft_pvp_bot_pure_python.zip
├── pvp_bot_tensorboard/        # Training logs
└── screenshots/                # Captured test images
```

## 📝 To-Do

- [ ] Add pre-trained models
- [ ] Create Docker container for easy setup
- [ ] Add video recording of training
- [ ] Implement curriculum learning framework
- [ ] Add support for servers with anti-cheat
- [ ] Create tournament mode for bot vs bot
- [ ] Add web dashboard for training

## 🎓 Learning Resources

**Reinforcement Learning:**
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## ⚠️ Disclaimer

This bot is for **educational purposes** and **single-player/private server use only**. 

- Do NOT use on public servers without permission
- May violate server rules or terms of service
- Use responsibly and ethically
- Author is not responsible for misuse

## 🙏 Acknowledgments

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms
- [OpenAI Gym](https://github.com/openai/gym) - RL environment framework
- [MSS](https://github.com/BoboTiG/python-mss) - Fast screen capture
- [PyDirectInput](https://github.com/learncodebygaming/pydirectinput) - Input simulation
