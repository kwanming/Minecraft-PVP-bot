"""
Helper script to find your Minecraft window position for screen capture
Run this while Minecraft is open to get the correct coordinates
"""

import cv2
import numpy as np
from mss import mss
from PIL import Image
import time

def capture_and_show():
    """Capture screen and help you identify Minecraft window"""
    
    with mss() as sct:
        # Get all monitors
        print("Available monitors:")
        for i, monitor in enumerate(sct.monitors):
            if i == 0:  # Skip "all monitors"
                continue
            print(f"  Monitor {i}: {monitor}")
        
        # Capture primary monitor
        monitor = sct.monitors[1]
        print(f"\nCapturing primary monitor: {monitor['width']}x{monitor['height']}")
        
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Save full screenshot
        cv2.imwrite('full_screen.png', img)
        print("✅ Saved 'full_screen.png' - open it to see your entire screen")
        
        # Now let's try common Minecraft window sizes
        common_sizes = [
            (854, 480),   # Default windowed
            (1280, 720),  # 720p
            (1920, 1080), # 1080p
            (1024, 768),  # Old standard
        ]
        
        print("\n" + "="*60)
        print("TESTING COMMON MINECRAFT WINDOW POSITIONS")
        print("="*60)
        
        for width, height in common_sizes:
            # Try centered position
            region = {
                'top': (monitor['height'] - height) // 2,
                'left': (monitor['width'] - width) // 2,
                'width': width,
                'height': height
            }
            
            screenshot = sct.grab(region)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            filename = f'test_{width}x{height}.png'
            cv2.imwrite(filename, img)
            print(f"\n📸 Saved '{filename}'")
            print(f"   Region: {region}")
            
        print("\n" + "="*60)
        print("WHAT TO DO NEXT:")
        print("="*60)
        print("1. Open the saved images (test_*.png)")
        print("2. Find which one shows your Minecraft window")
        print("3. Use those coordinates in your training script")
        print("\nExample usage:")
        print("  capture_region = {")
        print("      'top': 300,")
        print("      'left': 533,")
        print("      'width': 854,")
        print("      'height': 480")
        print("  }")
        print("  env = MinecraftPvPEnv(capture_region=capture_region)")
        print("="*60)


def interactive_capture():
    """Real-time interactive window finder"""
    print("\n🎮 INTERACTIVE MINECRAFT WINDOW FINDER")
    print("="*60)
    print("Instructions:")
    print("1. Make sure Minecraft is open and visible")
    print("2. This will show live preview - adjust with keyboard")
    print("3. Press 'q' when the box matches your Minecraft window")
    print("="*60 + "\n")
    
    input("Press Enter when ready...")
    
    with mss() as sct:
        monitor = sct.monitors[1]
        
        # Start with default centered region
        width, height = 854, 480
        top = (monitor['height'] - height) // 2
        left = (monitor['width'] - width) // 2
        
        print("\nControls:")
        print("  w/a/s/d: Move window (up/left/down/right)")
        print("  +/-: Resize (bigger/smaller)")
        print("  q: Quit and save settings")
        print("  c: Take screenshot")
        print("  r: Reset to center\n")
        
        cv2.namedWindow('Minecraft Window Finder', cv2.WINDOW_NORMAL)
        
        while True:
            region = {'top': top, 'left': left, 'width': width, 'height': height}
            
            # Capture
            screenshot = sct.grab(region)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Resize for display if too large
            display_img = cv2.resize(img, (854, 480)) if width > 854 else img.copy()
            
            # Add text overlay with background for readability
            overlay = display_img.copy()
            cv2.rectangle(overlay, (0, 0), (display_img.shape[1], 100), (0, 0, 0), -1)
            display_img = cv2.addWeighted(display_img, 0.7, overlay, 0.3, 0)
            
            cv2.putText(display_img, f"Position: ({left}, {top}) Size: {width}x{height}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_img, "w/a/s/d=move  +/-=resize  q=save  c=screenshot", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Minecraft Window Finder', display_img)
            
            key = cv2.waitKey(100) & 0xFF
            
            # Movement controls (WASD)
            if key == ord('w'):  # Up
                top = max(0, top - 20)
                print(f"↑ Moved up: top={top}")
            elif key == ord('s'):  # Down
                top = min(monitor['height'] - height, top + 20)
                print(f"↓ Moved down: top={top}")
            elif key == ord('a'):  # Left
                left = max(0, left - 20)
                print(f"← Moved left: left={left}")
            elif key == ord('d'):  # Right
                left = min(monitor['width'] - width, left + 20)
                print(f"→ Moved right: left={left}")
            
            # Resize controls
            elif key == ord('+') or key == ord('='):
                width = min(monitor['width'], width + 50)
                height = min(monitor['height'], height + 50)
                print(f"🔍 Increased size: {width}x{height}")
            elif key == ord('-') or key == ord('_'):
                width = max(100, width - 50)
                height = max(100, height - 50)
                print(f"🔍 Decreased size: {width}x{height}")
            
            # Reset to center
            elif key == ord('r'):
                width, height = 854, 480
                top = (monitor['height'] - height) // 2
                left = (monitor['width'] - width) // 2
                print("↻ Reset to center")
            
            # Screenshot
            elif key == ord('c'):
                cv2.imwrite('minecraft_capture_test.png', img)
                print("📸 Saved 'minecraft_capture_test.png'")
            
            # Quit and save
            elif key == ord('q'):
                print(f"\n✅ Final region: {region}")
                print("\n" + "="*60)
                print("ADD THIS TO YOUR CODE:")
                print("="*60)
                print(f"capture_region = {region}")
                print("\nExample usage:")
                print("env = MinecraftPvPEnv(capture_region=capture_region)")
                print("="*60 + "\n")
                break
        
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("MINECRAFT WINDOW FINDER")
    print("="*60)
    print("Choose mode:")
    print("1. Quick capture (saves test images)")
    print("2. Interactive finder (live preview)")
    choice = input("\nEnter 1 or 2: ").strip()
    
    if choice == "1":
        capture_and_show()
    elif choice == "2":
        interactive_capture()
    else:
        print("Invalid choice!")