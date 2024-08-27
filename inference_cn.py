import cv2
import numpy as np
import subprocess
import torch



def rgbnp2tensor(rgbnplist):
    rgbnps = np.array(rgbnplist).copy()
    lqinput = np.array(np.array(rgbnps)/255.0, np.float32) # [t,h,w,c]
    lqinput = torch.from_numpy(lqinput).permute(0,3,1,2).cuda()
    return lqinput


def apply_net_to_frames(frames,model,w=1.0):
    lqinput = rgbnp2tensor(frames)
    with torch.no_grad():
        restored_faces = model(lqinput,w=w)[0][1]
    restored_faces = torch.clamp(restored_faces,0,1)
    restored_face = restored_faces.detach().cpu().permute(1,2,0).numpy()*255
    npface = np.array(restored_face,np.uint8)
    return npface

def process_video_ffmpeg(input_path, output_path,model):
    # 使用FFmpeg打开输入视频，通过管道读取
    ffmpeg_input = [
        'ffmpeg_lib/ffmpeg', '-i', input_path,
        '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-'
    ]
    pipe_in = subprocess.Popen(ffmpeg_input, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 使用FFmpeg通过管道写入输出视频
    ffmpeg_output = [
        'ffmpeg_lib/ffmpeg', '-y',  # '-y' 表示直接覆盖输出文件
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps),
        '-i', '-', '-an', '-vcodec', 'libx265', '-crf', '18', '-tag:v', 'hvc1', output_path
    ]
    pipe_out = subprocess.Popen(ffmpeg_output, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # 初始化三帧的滑动窗口
    frame_buffer = []

    # 读取第一帧
    raw_frame = pipe_in.stdout.read(width * height * 3)
    if raw_frame:
        first_frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
        frame_buffer.append(first_frame)
        frame_buffer.append(first_frame)  # pad前一帧（复制第一帧）
    start = True
    while True:
        
        # 从管道读取一帧视频
        raw_frame = pipe_in.stdout.read(width * height * 3)
        if not raw_frame:
            break
        
        # 将原始帧转换为NumPy数组
        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
        
        # 将当前帧添加到缓冲区
        frame_buffer.append(frame)
        
        # 如果缓冲区中有三帧，进行处理
        if len(frame_buffer) == 3:
            # 对三帧进行处理（此处为示例，实际处理逻辑可以复杂化）
            processed_frame = apply_net_to_frames(frame_buffer,model)  
            
            # 将处理后的帧写入输出管道
            pipe_out.stdin.write(processed_frame.tobytes())
            
            # 移除缓冲区中的第一帧，继续处理
            frame_buffer.pop(0)

    # 处理最后的两帧（需要pad一帧的情况）
    if len(frame_buffer) == 2:
        frame_buffer.append(frame_buffer[-1])  # pad最后一帧
        processed_frame = apply_net_to_frames(frame_buffer,model)
        pipe_out.stdin.write(processed_frame.tobytes())
        
    # 关闭管道
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
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="裁剪视频的左侧部分")

    # 添加输入视频参数，默认值为 "assets/inputdemovideo.mp4"
    parser.add_argument(
        "-i", "--input_video", 
        type=str, 
        default="assets/inputdemovideo.mp4", 
        help="输入视频文件路径"
    )

    # 添加输出视频参数，默认值为 "exp/output_demo.mp4"
    parser.add_argument(
        "-o", "--output_video", 
        type=str, 
        default="exp/output_demo.mp4", 
        help="输出视频文件路径"
    )

    # 解析参数
    args = parser.parse_args()
    
    # 获取输入视频的基本信息（宽度、高度、帧率）
    cap = cv2.VideoCapture(args.input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    
    
    # 加载模型
    model = load_architecture()
    
    
    
    
    process_video_ffmpeg(args.input_video, args.output_video,model)
