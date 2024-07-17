import torch
from glob import glob
import os
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
from argparse import ArgumentParser


from models.models import DGModel_final
from utils.misc import denormalize, divide_img_into_patches, get_padding


@torch.no_grad()
def predict(model, img, patch_size=3584, log_para=1000):
   h, w = img.shape[2:]
   ps = patch_size
   if h >= ps or w >= ps:
       pred_dmap = torch.zeros(1, 1, h, w)
       pred_count = 0
       img_patches, nh, nw = divide_img_into_patches(img, ps)
       for i in range(nh):
           for j in range(nw):
               patch = img_patches[i*nw+j]
               pred_dpatch = model(patch)[0]
               pred_dmap[:, :, i*ps:(i+1)*ps, j*ps:(j+1)*ps] = pred_dpatch
   else:
       pred_dmap = model(img)[0]
   pred_count = pred_dmap.sum().cpu().item() / log_para


   return pred_dmap.squeeze().cpu().numpy(), pred_count


import cv2


def load_imgs(img_path, unit_size, device):
   imgs = []
   img_names = []


   assert os.path.exists(img_path), f'Video file {img_path} does not exist.'


   # Open the video file
   cap = cv2.VideoCapture(img_path)
   frame_idx = 0


   while cap.isOpened():
       ret, frame = cap.read()
       if not ret:
           break
      
       # Convert frame to PIL Image format
       frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       img = Image.fromarray(frame)


       if unit_size > 0:
           w, h = img.size
           new_w = (w // unit_size + 1) * unit_size if w % unit_size != 0 else w
           new_h = (h // unit_size + 1) * unit_size if h % unit_size != 0 else h


           padding, h, w = get_padding(h, w, new_h, new_w)


           img = F.pad(img, padding)
       img = F.to_tensor(img)
       img = F.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
       img = img.unsqueeze(0).to(device)
       imgs.append(img)
       img_names.append(f'frame_{frame_idx}.jpg')  # Or any naming convention you prefer
       frame_idx += 1
  
   cap.release()


   return imgs, img_names


def load_model(model_path, device):
   model_file = os.path.join(model_path, 'best.pth')  # Adjust 'model.pth' to your actual model file name
   device = torch.device('cpu')  # Force model to be loaded on CPU
   model = DGModel_final().to(device)
   model.load_state_dict(torch.load(model_file, map_location=device), strict=False)
   model.eval()


   return model


def animate_results(filename, vis_dir):
   frames = []
   values = []


   # Read data from the file
   with open(filename, 'r') as file:
       lines = file.readlines()
       for line in lines:
           parts = line.strip().split(':')
           if len(parts) >= 2:
               try:
                   frame = int(parts[0].strip().split()[1])
                   value = float(parts[1].strip())
                   frames.append(frame)
                   values.append(value)
               except (IndexError, ValueError) as e:
                   print(f"Error processing line: {line.strip()}. Skipping.")


   # Set up the figure and axis
   fig, ax = plt.subplots()
   line, = ax.plot([], [], lw=2)
   ax.set_xlim(0, len(frames))  # Set x-axis limit based on number of frames
   ax.set_ylim(min(values) - 5, max(values) + 5 if values else 1)
   ax.set_xlabel('Frame')
   ax.set_ylabel('Value')
   ax.set_title('Moving Line Graph')


   # Initialize the plot
   def init():
       line.set_data([], [])
       return line,


   # Update function
   def update(frame):
       # Calculate the new x and y data
       xdata = frames[:frame+1]  # Incrementally add frames
       ydata = values[:frame+1]
      
       # Adjust xlim to keep the window moving
       ax.set_xlim(max(0, frame - 100), max(len(frames), frame + 1))
      
       line.set_data(xdata, ydata)
       return line,


   # Create animation
   ani = animation.FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=True, interval=100)


   # Save animation as video
   animation_path = os.path.join(vis_dir, 'animation.mp4')
   ani.save(animation_path, writer='ffmpeg')


   print(f"Animation saved as {animation_path}")


import numpy as np


def main(args):
   # Extract the base name of the input file without extension
   input_base_name = os.path.splitext(os.path.basename(args.img_path))[0]
   output_folder = f'{input_base_name}_output'
   os.makedirs(output_folder, exist_ok=True)


   imgs, img_names = load_imgs(args.img_path, args.unit_size, args.device)
   model = load_model(args.model_path, args.device)


   save_path = os.path.join(output_folder, args.save_path) if args.save_path is not None else None
   vis_dir = os.path.join(output_folder, args.vis_dir) if args.vis_dir is not None else None


   start_time = time()
   for idx, (img, img_name) in enumerate(zip(imgs, img_names)):
       pred_dmap, pred_count = predict(model, img, args.patch_size, args.log_para)
       print(f'Frame {idx}: {pred_count}')


       if save_path is not None:
           with open(save_path, 'a') as f:
               f.write(f'Frame {idx}: {pred_count}\n')


       if vis_dir is not None:
           os.makedirs(vis_dir, exist_ok=True)
           denormed_img = denormalize(img)[0].cpu().permute(1, 2, 0).numpy()
           vis_name = f'frame_{idx}.png'  # Unique filename for each frame
           vis_path = os.path.join(vis_dir, vis_name)
           plt.figure(figsize=(10, 5))
           plt.imshow(denormed_img)
           plt.title(f'Frame {idx} - Predicted count: {pred_count}')
           plt.savefig(vis_path)
           plt.close()


   print(f'Total time: {time()-start_time:.2f}s')


   # Call the animation function
   if save_path is not None:
       animate_results(save_path, vis_dir)


if __name__ == '__main__':
   parser = ArgumentParser()
   parser.add_argument('--img_path', type=str, required=True, help='Path to the image or directory containing images.')
   parser.add_argument('--model_path', type=str, required=True, help='Path to the model weight.')
   parser.add_argument('--save_path', type=str, default=None, help='Path of the text file to save the prediction results.')
   parser.add_argument('--vis_dir', type=str, default=None, help='Directory to save the visualization results.')
   parser.add_argument('--unit_size', type=int, default=16, help='Unit size for image resizing. Normally set to 16 and no need to change.')
   parser.add_argument('--patch_size', type=int, default=3584, help='Patch size for image division. Decrease this value if OOM occurs.')
   parser.add_argument('--log_para', type=int, default=1000, help='Parameter for log transformation. Normally set to 1000 and no need to change.')
   parser.add_argument('--device', type=str, default='cuda', help='Device to run the model. Default is cuda.')
   args = parser.parse_args()


   main(args)

