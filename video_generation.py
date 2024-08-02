import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# Load the pipeline
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# Generate the video frames
prompt = "Spiderman is surfing"
video_frames = pipe(prompt, num_inference_steps=25).frames

# Flatten the list of frames
flattened_frames = [frame for sublist in video_frames for frame in sublist]

# Export the flattened frames to a video
try:
    video_path = export_to_video(flattened_frames)
    print(f"Video exported to {video_path}")
except ValueError as e:
    print(f"Error exporting video: {e}")



