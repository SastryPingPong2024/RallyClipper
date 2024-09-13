import os
import argparse
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from inference import *

def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    
    # Parse the output and convert to integers
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory

def set_available_gpu():
    try:
        gpu_memory_map = get_gpu_memory_map()
        if gpu_memory_map:
            # Select the GPU with the most free memory
            selected_gpu = gpu_memory_map.index(max(gpu_memory_map))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
            return selected_gpu
        else:
            print("No GPUs available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("nvidia-smi is not available. Make sure CUDA is installed and you have NVIDIA GPUs.")
            
def process_video(video_file, args):
    gpu_id = set_available_gpu()
    print(f"Processing {video_file} on GPU {gpu_id}")
    video_path = os.path.join(args.root_dir, video_file)
    model = CNNVideoFrameClassifier(args.width, args.height)
    model.load_state_dict(torch.load('best_model.pth', map_location=f'cuda'))
    run_inference(model, video_path, args.output_dir, args.width, args.height, args.batch_size, 0)
    print(f"Finished processing {video_file}")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True) # Ensure output folder exists
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']  # Add or remove extensions as needed
    video_files = [f for f in os.listdir(args.root_dir) if os.path.splitext(f)[1].lower() in video_extensions]
    if not video_files:
        raise ValueError(f"No video files found in {args.root_dir}")
    else:
        print(f"Found {len(video_files)} video files.")

    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = []
        for video_file in video_files:
            future = executor.submit(process_video, video_file, args)
            futures.append(future)
            time.sleep(10)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Run inference on videos in a folder using a trained model.")
    parser.add_argument("root_dir", type=str, help="Path to the folder containing input video files")
    parser.add_argument("output_dir", type=str, help="Folder to save the output video segments")
    parser.add_argument("--width", type=int, default=WIDTH, help="Frame width for model input")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Frame height for model input")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()
    main(args)
