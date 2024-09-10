import os
import cv2
import numpy as np
import subprocess
import torch

def rgbnp2tensor(rgbnplist):
    rgbnps = np.array(rgbnplist).copy()
    lqinput = np.array(np.array(rgbnps) / 255.0, np.float32)  # [t,h,w,c]
    lqinput = torch.from_numpy(lqinput).permute(0, 3, 1, 2).cuda()
    return lqinput

def apply_net_to_frames(frames, model, w=1.0):
    lqinput = rgbnp2tensor(frames)
    with torch.no_grad():
        restored_faces = model(lqinput, w=w)[0][1]
    restored_faces = torch.clamp(restored_faces, 0, 1)
    restored_face = restored_faces.detach().cpu().permute(1, 2, 0).numpy() * 255
    npface = np.array(restored_face, np.uint8)
    return npface

def process_video_save_frames(input_path, output_folder, model):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Use OpenCV to read the input video
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize a sliding window of three frames
    frame_buffer = []
    frame_count = 0

    # Read the first frame
    ret, first_frame = cap.read()
    if ret:
        frame_buffer.append(first_frame)
        frame_buffer.append(first_frame)  # Pad the previous frame (duplicate the first frame)

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add the current frame to the buffer
        frame_buffer.append(frame)
        
        # If there are three frames in the buffer, process them
        if len(frame_buffer) == 3:
            # Process the three frames
            processed_frame = apply_net_to_frames(frame_buffer, model)
            
            # Save the processed frame to the output folder
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, processed_frame)
            frame_count += 1
            
            # Remove the first frame from the buffer and continue
            frame_buffer.pop(0)

    # Process the last two frames (when padding a frame is needed)
    if len(frame_buffer) == 2:
        frame_buffer.append(frame_buffer[-1])  # Pad the last frame
        processed_frame = apply_net_to_frames(frame_buffer, model)
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, processed_frame)
    
    # Release the video capture
    cap.release()

def create_video_from_frames(output_folder, output_video, fps, width, height):
    # Use FFmpeg to create a video from the saved frames
    ffmpeg_command = [
        'ffmpeg_lib/ffmpeg', '-y',  # '-y' means overwrite the output file
        '-framerate', str(fps), 
        '-i', os.path.join(output_folder, 'frame_%04d.png'),  # Input frames pattern
        '-s', f'{width}x{height}',  # Frame size
        '-vcodec', 'libx265', '-crf', '18', '-tag:v', 'hvc1',  # Encoding options
        output_video
    ]
    
    subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def load_architecture(weights='weights/weights.pth'):
    from archs.pgtformer_arch import PGTFormer
    import yaml
    with open('options/release_test_stage_IIII_dont_need_align_version.yml', mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    ooo = opt['network_g']
    # network = PGTFormer(**ooo).cuda()
    # state_dict = torch.load(weights)
    # network.load_state_dict(state_dict=state_dict['params_ema'])
    model = PGTFormer.from_pretrained("kepeng/pgtformer-base").cuda()
    model.eval()
    model.requires_grad_(False)
    return model

if __name__ == "__main__":
    import argparse
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Process video frames and save to folder")

    # Add input video argument, default to "assets/inputdemovideo.mp4"
    parser.add_argument(
        "-i", "--input_video", 
        type=str, 
        default="assets/inputdemovideo.mp4", 
        help="Input video file path"
    )

    # Add output folder for frames, default to "exp/frames"
    parser.add_argument(
        "-f", "--output_folder", 
        type=str, 
        default="exp/frames", 
        help="Output folder for frames"
    )

    # Add output video argument, default to "exp/output_demo.mp4"
    parser.add_argument(
        "-o", "--output_video", 
        type=str, 
        default="exp/output_demo.mp4", 
        help="Output video file path"
    )

    # Parse arguments
    args = parser.parse_args()
    
    # Get basic information about the input video (width, height, frame rate)
    cap = cv2.VideoCapture(args.input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Load the model
    model = load_architecture()
    
    # Process the video and save frames
    process_video_save_frames(args.input_video, args.output_folder, model)
    
    # Create a video from the saved frames
    create_video_from_frames(args.output_folder, args.output_video, fps, width, height)
