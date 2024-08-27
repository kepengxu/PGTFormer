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

def process_video_ffmpeg(input_path, output_path, model):
    # Use FFmpeg to open the input video and read it via a pipe
    ffmpeg_input = [
        'ffmpeg_lib/ffmpeg', '-i', input_path,
        '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-'
    ]
    pipe_in = subprocess.Popen(ffmpeg_input, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Use FFmpeg to write the output video via a pipe
    ffmpeg_output = [
        'ffmpeg_lib/ffmpeg', '-y',  # '-y' means overwrite the output file
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps),
        '-i', '-', '-an', '-vcodec', 'libx265', '-crf', '18', '-tag:v', 'hvc1', output_path
    ]
    pipe_out = subprocess.Popen(ffmpeg_output, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Initialize a sliding window of three frames
    frame_buffer = []

    # Read the first frame
    raw_frame = pipe_in.stdout.read(width * height * 3)
    if raw_frame:
        first_frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
        frame_buffer.append(first_frame)
        frame_buffer.append(first_frame)  # Pad the previous frame (duplicate the first frame)
    start = True
    while True:
        # Read a frame from the pipe
        raw_frame = pipe_in.stdout.read(width * height * 3)
        if not raw_frame:
            break
        
        # Convert the raw frame to a NumPy array
        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
        
        # Add the current frame to the buffer
        frame_buffer.append(frame)
        
        # If there are three frames in the buffer, process them
        if len(frame_buffer) == 3:
            # Process the three frames (this is an example; actual logic may vary)
            processed_frame = apply_net_to_frames(frame_buffer, model)
            
            # Write the processed frame to the output pipe
            pipe_out.stdin.write(processed_frame.tobytes())
            
            # Remove the first frame from the buffer and continue
            frame_buffer.pop(0)

    # Process the last two frames (when padding a frame is needed)
    if len(frame_buffer) == 2:
        frame_buffer.append(frame_buffer[-1])  # Pad the last frame
        processed_frame = apply_net_to_frames(frame_buffer, model)
        pipe_out.stdin.write(processed_frame.tobytes())
        
    # Close the pipes
    pipe_in.stdout.close()
    pipe_out.stdin.close()
    pipe_in.wait()
    pipe_out.wait()
    
import yaml
from collections import OrderedDict

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def load_architecture(weights='weights/weights.pth'):
    from archs.pgtformer_arch import PGTFormer
    import yaml
    with open('options/release_test_stage_IIII_dont_need_align_version.yml', mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    ooo = opt['network_g']
    network = PGTFormer(**ooo).cuda()
    state_dict = torch.load(weights)
    network.load_state_dict(state_dict=state_dict['params_ema'])
    network.eval()
    network.requires_grad_(False)
    return network

if __name__ == "__main__":
    import argparse
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Crop the left part of a video")

    # Add input video argument, default to "assets/inputdemovideo.mp4"
    parser.add_argument(
        "-i", "--input_video", 
        type=str, 
        default="assets/inputdemovideo.mp4", 
        help="Input video file path"
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
    
    process_video_ffmpeg(args.input_video, args.output_video, model)
