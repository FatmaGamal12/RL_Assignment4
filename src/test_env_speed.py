"""
Quick diagnostic to test CarRacing environment speed.
Run this to identify if environments are the bottleneck.
"""

import time
import gymnasium as gym
import numpy as np

def test_single_env():
    """Test single environment initialization and stepping."""
    print("=" * 60)
    print("Testing Single CarRacing-v3 Environment")
    print("=" * 60)
    
    print("\n1. Creating environment...", flush=True)
    start = time.time()
    env = gym.make("CarRacing-v3", continuous=True, render_mode=None)
    print(f"   ✓ Created in {time.time() - start:.2f}s")
    
    print("\n2. First reset (slowest)...", flush=True)
    start = time.time()
    obs, info = env.reset()
    print(f"   ✓ Reset in {time.time() - start:.2f}s")
    print(f"   Observation shape: {obs.shape}")
    
    print("\n3. Second reset (should be faster)...", flush=True)
    start = time.time()
    obs, info = env.reset()
    print(f"   ✓ Reset in {time.time() - start:.2f}s")
    
    print("\n4. Taking 100 random steps...", flush=True)
    start = time.time()
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, info = env.reset()
    elapsed = time.time() - start
    fps = 100 / elapsed
    print(f"   ✓ Completed in {elapsed:.2f}s ({fps:.1f} FPS)")
    
    env.close()
    print("\n✓ Single environment test passed!")


def test_parallel_envs(n_envs=8):
    """Test parallel environment creation."""
    print("\n" + "=" * 60)
    print(f"Testing {n_envs} Parallel CarRacing-v3 Environments")
    print("=" * 60)
    
    print(f"\n1. Creating {n_envs} environments...", flush=True)
    envs = []
    start = time.time()
    
    for i in range(n_envs):
        print(f"   Creating env {i+1}/{n_envs}...", end='\r', flush=True)
        env = gym.make("CarRacing-v3", continuous=True, render_mode=None)
        envs.append(env)
    
    print(f"\n   ✓ Created {n_envs} envs in {time.time() - start:.2f}s")
    
    print(f"\n2. Resetting all {n_envs} environments...", flush=True)
    start = time.time()
    for i, env in enumerate(envs):
        print(f"   Resetting env {i+1}/{n_envs}...", end='\r', flush=True)
        obs, info = env.reset()
    elapsed = time.time() - start
    print(f"\n   ✓ Reset all in {elapsed:.2f}s ({elapsed/n_envs:.2f}s per env)")
    
    print(f"\n3. Taking 50 steps in all environments...", flush=True)
    start = time.time()
    total_steps = 0
    for step in range(50):
        for i, env in enumerate(envs):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_steps += 1
            if done or truncated:
                obs, info = env.reset()
    
    elapsed = time.time() - start
    fps = total_steps / elapsed
    print(f"   ✓ Completed {total_steps} steps in {elapsed:.2f}s ({fps:.1f} FPS)")
    
    # Cleanup
    for env in envs:
        env.close()
    
    print(f"\n✓ Parallel environments test passed!")


def test_preprocessing():
    """Test preprocessing speed."""
    print("\n" + "=" * 60)
    print("Testing Image Preprocessing Speed")
    print("=" * 60)
    
    from PIL import Image
    
    # Create dummy frame
    frame = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    
    print("\n1. Testing cropping + grayscale + resize...", flush=True)
    start = time.time()
    
    for i in range(1000):
        # Crop
        cropped = frame[:84, 6:90, :]
        # Grayscale
        gray = 0.299 * cropped[:, :, 0] + 0.587 * cropped[:, :, 1] + 0.114 * cropped[:, :, 2]
        # Resize
        img = Image.fromarray(gray.astype(np.uint8))
        img = img.resize((64, 64), Image.BILINEAR)
        processed = np.array(img, dtype=np.float32) / 255.0
    
    elapsed = time.time() - start
    per_frame = elapsed / 1000 * 1000  # ms per frame
    print(f"   ✓ Processed 1000 frames in {elapsed:.2f}s ({per_frame:.2f}ms per frame)")
    print(f"   Theoretical max FPS: {1000/elapsed:.1f}")


if __name__ == "__main__":
    print("\n🔍 CarRacing-v3 Environment Diagnostics\n")
    
    try:
        test_single_env()
        test_parallel_envs(n_envs=8)
        test_preprocessing()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nIf this completes quickly but training hangs,")
        print("the issue is likely in the PPO training loop.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()