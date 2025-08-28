import torch
from ultralytics import YOLO
import os
import tkinter as tk
from tkinter import filedialog
import shutil

def select_video_file():
    """Select video file using file dialog"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select video file",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def main():
    print("Please select your video file...")
    video_path = select_video_file()
    
    if not video_path:
        print("No file selected. Exiting...")
        return
    
    print(f"Selected video: {video_path}")
    
    
    model_path = 'best.pt' 
    
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found!")
        print("Please make sure 'best.pt' is in the same directory as this script.")
        return
    
    print("Loading YOLO model...")
    model = YOLO(model_path)
    
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    
    # Process video
    print("Processing video...")
    results = model.predict(
        source=video_path,
        device=device,
        save=True,
        conf=0.5,
        show=False,
        project='runs',  # Output directory
        name='detect'    # Subdirectory name
    )
    
    # Find and move the output video to current directory
    output_base_dir = 'runs/detect'
    
    # Find the latest predict directory
    predict_dirs = [d for d in os.listdir(output_base_dir) if d.startswith('detect')]
    if predict_dirs:
        # Sort by creation time to get the latest
        predict_dirs.sort(key=lambda x: os.path.getctime(os.path.join(output_base_dir, x)), reverse=True)
        latest_dir = os.path.join(output_base_dir, predict_dirs[0])
        
        # Find output video file
        for file in os.listdir(latest_dir):
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')):
                source_path = os.path.join(latest_dir, file)
                destination_path = f"output_{file}"
                
                # Copy to current directory
                shutil.copy2(source_path, destination_path)
                print(f"Output video saved as: {destination_path}")
                
                # Open the output directory
                try:
                    if os.name == 'nt':  # Windows
                        os.startfile(os.path.dirname(os.path.abspath(destination_path)))
                    elif os.name == 'posix':  # macOS and Linux
                        os.system(f'open "{os.path.dirname(os.path.abspath(destination_path))}"')
                except:
                    print(f"Output saved in: {os.path.abspath(destination_path)}")
                
                break
        else:
            print("No output video found!")
    else:
        print("No output directory found!")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()