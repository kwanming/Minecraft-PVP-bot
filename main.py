import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import cv2
from mss import mss
import time
import pydirectinput
from pynput import keyboard

# Custom CNN Feature Extractor
class MinecraftCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class MinecraftController:
    """
    Direct keyboard/mouse control of Minecraft
    """
    
    def __init__(self):
        pydirectinput.FAILSAFE = False
        self.keys_pressed = set()
        self.center_x = 1920 // 2
        self.center_y = 1080 // 2
        
    def set_window_center(self, capture_region):
        """Set the center of Minecraft window for mouse control"""
        self.center_x = capture_region['left'] + capture_region['width'] // 2
        self.center_y = capture_region['top'] + capture_region['height'] // 2
        print(f"🎯 Mouse center set to: ({self.center_x}, {self.center_y})")
        
    def press_key(self, key, duration=0.05):
        """Press and hold a key"""
        pydirectinput.keyDown(key)
        time.sleep(duration)
        pydirectinput.keyUp(key)
    
    def move_mouse(self, dx, dy):
        """Move mouse relative to current position"""
        pydirectinput.moveRel(dx, dy, relative=True)
    
    def look_horizontal(self, direction, amount=50):
        """Look left or right"""
        self.move_mouse(direction * amount, 0)
    
    def look_vertical(self, direction, amount=30):
        """Look up or down"""
        self.move_mouse(0, direction * amount)
    
    def click(self):
        """Left click (attack)"""
        pydirectinput.click()
    
    def execute_action(self, action):
        """Execute action by simulating keyboard/mouse input"""
        # Release all previous keys
        for key in ['w', 'a', 's', 'd', 'space', 'shift', 'ctrl']:
            try:
                pydirectinput.keyUp(key)
            except:
                pass
        
        # Return True if action involves attacking
        is_attack_action = False
        
        if action == 0:  # No-op
            pass
        elif action == 1:  # Forward
            self.press_key('w')
        elif action == 2:  # Back
            self.press_key('s')
        elif action == 3:  # Left
            self.press_key('a')
        elif action == 4:  # Right
            self.press_key('d')
        elif action == 5:  # Attack
            self.click()
            is_attack_action = True
        elif action == 6:  # Jump
            self.press_key('space')
        elif action == 7:  # Sprint + Forward
            pydirectinput.keyDown('w')
            time.sleep(0.05)
            pydirectinput.keyUp('w')
        elif action == 8:  # Crouch
            self.press_key('shift')
        elif action == 9:  # Look left
            self.look_horizontal(-1, amount=40)
        elif action == 10:  # Look right
            self.look_horizontal(1, amount=40)
        elif action == 11:  # Look up
            self.look_vertical(-1, amount=30)
        elif action == 12:  # Look down
            self.look_vertical(1, amount=30)
        elif action == 13:  # Forward + Attack
            pydirectinput.keyDown('w')
            self.click()
            time.sleep(0.05)
            pydirectinput.keyUp('w')
            is_attack_action = True
        elif action == 14:  # Strafe left + Attack
            pydirectinput.keyDown('a')
            self.click()
            time.sleep(0.05)
            pydirectinput.keyUp('a')
            is_attack_action = True
        elif action == 15:  # Strafe right + Attack
            pydirectinput.keyDown('d')
            self.click()
            time.sleep(0.05)
            pydirectinput.keyUp('d')
            is_attack_action = True
        
        return is_attack_action


class MinecraftPvPEnv(gym.Env):
    """
    Pure Python Minecraft PvP Environment with Entity Hit Detection
    """
    
    def __init__(self, capture_region=None, render_mode=None):
        super().__init__()
        
        # Action space: 16 actions
        self.action_space = spaces.Discrete(16)
        
        # Observation: 84x84 RGB image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, 84, 84), dtype=np.uint8
        )
        
        self.render_mode = render_mode
        self.episode_steps = 0
        self.max_steps = 1000
        
        # Screen capture
        self.sct = mss()
        
        # Set capture region
        if capture_region is None:
            monitor = self.sct.monitors[1]
            width, height = 854, 480
            self.capture_region = {
                'top': (monitor['height'] - height) // 2,
                'left': (monitor['width'] - width) // 2,
                'width': width,
                'height': height
            }
            print(f"⚠️  Using default capture region: {self.capture_region}")
        else:
            self.capture_region = capture_region
        
        # Minecraft controller
        self.controller = MinecraftController()
        self.controller.set_window_center(self.capture_region)
        
        # Health tracking
        self.last_health_estimate = 1.0
        self.damage_dealt = 0
        self.damage_taken = 0
        
        # **NEW: Hit detection tracking**
        self.last_screen = None
        self.last_entity_flash = 0  # Time of last entity hit flash
        self.total_hits = 0
        self.total_misses = 0
        self.consecutive_misses = 0
        self.attack_cooldown = 0  # Frames since last attack
        
        print("\n" + "="*60)
        print("🎮 MINECRAFT PVP BOT WITH HIT DETECTION!")
        print("="*60)
        print(f"📹 Capturing screen: {self.capture_region}")
        print("⚔️  Hit detection: ENABLED")
        print("❌ Miss punishment: ENABLED")
        print("="*60 + "\n")
    
    def _capture_screen(self):
        """Capture Minecraft screen"""
        try:
            screenshot = self.sct.grab(self.capture_region)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            return img
        except Exception as e:
            print(f"❌ Screen capture failed: {e}")
            return np.zeros((self.capture_region['height'], 
                           self.capture_region['width'], 3), dtype=np.uint8)
    
    def _detect_entity_hit(self, current_screen, previous_screen):
        """
        Detect if an entity was hit by looking for:
        1. Red flash on entity (damage indicator)
        2. Entity shake/knockback effect
        3. Center screen flash (crosshair hit indicator)
        
        Returns: True if hit detected, False otherwise
        """
        if previous_screen is None:
            return False
        
        # Method 1: Detect red damage flash in center region
        center_h, center_w = current_screen.shape[0] // 2, current_screen.shape[1] // 2
        center_region_curr = current_screen[
            center_h - 50:center_h + 50,
            center_w - 50:center_w + 50
        ]
        center_region_prev = previous_screen[
            center_h - 50:center_h + 50,
            center_w - 50:center_w + 50
        ]
        
        # Calculate red channel difference
        red_diff = np.mean(center_region_curr[:, :, 0]) - np.mean(center_region_prev[:, :, 0])
        
        # If red channel increased significantly, entity was hit
        if red_diff > 15:  # Threshold for red flash detection
            return True
        
        # Method 2: Detect overall screen flash (hit marker)
        brightness_curr = np.mean(current_screen)
        brightness_prev = np.mean(previous_screen)
        brightness_diff = brightness_curr - brightness_prev
        
        if brightness_diff > 10:  # Screen flash from hit
            return True
        
        # Method 3: Detect entity shake (sudden color change in entity area)
        diff = cv2.absdiff(current_screen, previous_screen)
        motion = np.sum(diff) / diff.size
        
        if motion > 20:  # High motion = entity knockback
            return True
        
        return False
    
    def _detect_crosshair_on_entity(self, screen):
        """
        Detect if crosshair is on an entity by analyzing center pixels
        Entities typically have distinct colors compared to background
        
        Returns: confidence score (0-1) that crosshair is on entity
        """
        h, w = screen.shape[0], screen.shape[1]
        center_x, center_y = w // 2, h // 2
        
        # Sample small region around crosshair
        crosshair_region = screen[
            center_y - 10:center_y + 10,
            center_x - 10:center_x + 10
        ]
        
        # Calculate color variance (entities have more varied colors than sky/ground)
        color_variance = np.var(crosshair_region)
        
        # Normalize to 0-1 range
        entity_confidence = min(1.0, color_variance / 1000)
        
        return entity_confidence
    
    def _estimate_health_from_screen(self, screen):
        """Estimate health by analyzing the health bar region"""
        bottom_region = screen[-50:, :200]
        red_channel = bottom_region[:, :, 0]
        red_pixels = np.sum(red_channel > 150)
        health_estimate = min(1.0, red_pixels / 5000)
        return health_estimate
    
    def _process_observation(self, screen):
        """Process screen to observation"""
        obs = cv2.resize(screen, (84, 84), interpolation=cv2.INTER_AREA)
        obs = np.transpose(obs, (2, 0, 1))
        return obs.astype(np.uint8)
    
    def _calculate_reward(self, screen, action_was_attack):
        """
        Calculate reward with hit/miss detection
        """
        reward = 0.0
        
        # Health-based rewards
        current_health = self._estimate_health_from_screen(screen)
        health_change = current_health - self.last_health_estimate
        
        if health_change > 0:
            reward += 20
        elif health_change < 0:
            reward -= abs(health_change) * 50
            self.damage_taken += abs(health_change)
        
        self.last_health_estimate = current_health
        
        # Death penalty
        if current_health < 0.1:
            reward -= 100
        
        # **NEW: Hit/Miss Detection Rewards**
        if action_was_attack and self.attack_cooldown == 0:
            # Check if entity was hit
            hit_detected = self._detect_entity_hit(screen, self.last_screen)
            
            if hit_detected:
                # **REWARD for successful hit**
                reward += 50  # Large reward for hitting
                self.total_hits += 1
                self.consecutive_misses = 0
                self.last_entity_flash = self.episode_steps
                print(f"✅ HIT! Total hits: {self.total_hits}")
            else:
                # **PUNISHMENT for missing attack**
                reward -= 10  # Penalty for missing
                self.total_misses += 1
                self.consecutive_misses += 1
                print(f"❌ MISS! Total misses: {self.total_misses}")
                
                # **EXTRA punishment for consecutive misses**
                if self.consecutive_misses >= 3:
                    reward -= 15  # Additional penalty
                    print(f"⚠️  {self.consecutive_misses} consecutive misses!")
            
            # Set attack cooldown to prevent spamming
            self.attack_cooldown = 5  # Wait 5 frames before next attack counts
        
        # **BONUS: Reward for aiming at entities (even without attacking)**
        entity_aim_confidence = self._detect_crosshair_on_entity(screen)
        if entity_aim_confidence > 0.5:
            reward += 2 * entity_aim_confidence  # Small reward for good aim
        
        # Survival bonus
        reward += 0.5
        
        # Health maintenance bonus
        reward += current_health * 0.5
        
        # Decrease attack cooldown
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
        
        return reward
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_steps = 0
        self.damage_dealt = 0
        self.damage_taken = 0
        self.total_hits = 0
        self.total_misses = 0
        self.consecutive_misses = 0
        self.attack_cooldown = 0
        self.last_screen = None
        
        # Respawn if dead
        time.sleep(0.5)
        pydirectinput.click()
        time.sleep(0.5)
        
        # Capture initial screen
        screen = self._capture_screen()
        self.last_health_estimate = self._estimate_health_from_screen(screen)
        self.last_screen = screen.copy()
        
        observation = self._process_observation(screen)
        info = {
            'health': self.last_health_estimate,
            'hits': self.total_hits,
            'misses': self.total_misses
        }
        
        return observation, info
    
    def step(self, action):
        self.episode_steps += 1
        
        # Execute action and check if it was an attack
        action_was_attack = self.controller.execute_action(action)
        
        # Small delay
        time.sleep(0.05)
        
        # Capture screen
        screen = self._capture_screen()
        
        # Process observation
        observation = self._process_observation(screen)
        
        # Calculate reward (now considers hits/misses)
        reward = self._calculate_reward(screen, action_was_attack)
        
        # Store current screen for next hit detection
        self.last_screen = screen.copy()
        
        # Check termination
        current_health = self.last_health_estimate
        terminated = current_health < 0.1
        truncated = self.episode_steps >= self.max_steps
        
        # Calculate hit accuracy
        total_attacks = self.total_hits + self.total_misses
        hit_accuracy = (self.total_hits / total_attacks * 100) if total_attacks > 0 else 0
        
        info = {
            'health': current_health,
            'episode_steps': self.episode_steps,
            'damage_taken': self.damage_taken,
            'hits': self.total_hits,
            'misses': self.total_misses,
            'hit_accuracy': hit_accuracy,
            'consecutive_misses': self.consecutive_misses
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            accuracy = (self.total_hits / max(1, self.total_hits + self.total_misses)) * 100
            print(f"Step: {self.episode_steps}, Health: {self.last_health_estimate:.2f}, "
                  f"Hits: {self.total_hits}, Misses: {self.total_misses}, Accuracy: {accuracy:.1f}%")
    
    def close(self):
        self.sct.close()


# Training Callback with Hit Tracking
class PvPTrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.best_reward = -float('inf')
        self.best_accuracy = 0
    
    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            self.episode_count += 1
            ep_reward = self.locals.get('rewards', [0])[0]
            info = self.locals['infos'][0]
            
            accuracy = info.get('hit_accuracy', 0)
            
            if ep_reward > self.best_reward:
                self.best_reward = ep_reward
                print(f"🏆 New best reward: {ep_reward:.2f}")
            
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                print(f"🎯 New best accuracy: {accuracy:.1f}%")
            
            if self.verbose > 0:
                print(f"\n{'='*50}")
                print(f"Episode {self.episode_count} finished!")
                print(f"Reward: {ep_reward:.2f}")
                print(f"Final Health: {info.get('health', 0):.2f}")
                print(f"Steps Survived: {info.get('episode_steps', 0)}")
                print(f"⚔️  Hits: {info.get('hits', 0)}")
                print(f"❌ Misses: {info.get('misses', 0)}")
                print(f"🎯 Hit Accuracy: {accuracy:.1f}%")
                print(f"{'='*50}\n")
        
        return True


def train_pvp_bot(capture_region=None, total_timesteps=100_000):
    """Train the PvP bot"""
    print("🚀 STARTING MINECRAFT PVP BOT WITH HIT DETECTION")
    print("\n⚠️  BEFORE YOU START:")
    print("1. Open Minecraft and join a world/server")
    print("2. Enable PvP and spawn mobs/enemies")
    print("3. Position Minecraft window correctly")
    print("4. DON'T touch keyboard/mouse during training!\n")
    
    input("Press Enter when ready to start training...")
    
    env = MinecraftPvPEnv(capture_region=capture_region)
    
    policy_kwargs = dict(
        features_extractor_class=MinecraftCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )
    
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./pvp_bot_tensorboard/",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    callback = PvPTrainingCallback(verbose=1)
    
    print("\n🎮 Training started! Watch hit detection in action!")
    print("Press Ctrl+C to stop training and save model\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    model.save("minecraft_pvp_bot_hit_detection")
    print("\n✅ Model saved as 'minecraft_pvp_bot_hit_detection'!")
    
    env.close()
    return model


def evaluate_bot(model_path="minecraft_pvp_bot_hit_detection", capture_region=None, num_episodes=5):
    """Evaluate trained bot"""
    print("🎯 Evaluating bot with hit detection...")
    
    env = MinecraftPvPEnv(capture_region=capture_region, render_mode="human")
    model = PPO.load(model_path)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        print(f"\n📺 Episode {episode + 1} started...")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            env.render()
        
        accuracy = info.get('hit_accuracy', 0)
        print(f"✅ Episode {episode + 1} - Reward: {episode_reward:.2f}, Accuracy: {accuracy:.1f}%")
    
    env.close()


if __name__ == "__main__":
    import sys
    
    capture_region = None
    
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate_bot(capture_region=capture_region)
    else:
        train_pvp_bot(capture_region=capture_region, total_timesteps=100_000)