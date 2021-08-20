from transporter import *
from transporter_orig import *
from pl_transporter import *
import argparse

if not os.path.exists('output'):
    os.mkdir('output')
parser = argparse.ArgumentParser(description='LUS keypoint network pytorch-lightning parallel')
parser.add_argument('--ckpt', type=str, default='ckpt_TPRv1_LUS_radonbpm_lr0.0001_LUS_nossim_dptdecay_samrate10_VER_HORIZ_weightedHlam_radonbpm_-epoch=59-val_loss=0.00012.ckpt', help='')
parser.add_argument('--img', type=str, default='', help='')
parser.add_argument('--vid', type=str, default='', help='')
parser.add_argument('--frame_no', type=int, default=-1, help='')
parser.add_argument('--device', type=str, default='cpu', choices = ['cuda','cpu'], help='')

args = parser.parse_args()
print(args)

def depth_decay_coeff(x_cur, x = 256):
  '''
  x_cur: current value of depth 
  x: maximum depth (i.e, for a square image, the dimension, i.e, the height/width)
  This function return the coeff. to be 1 for less than half the depth, then decays it from 1 to 0.1 linearly till the end
  '''
  #if x_cur < x//2:
  if x_cur < 2*x//5:
    return 1
  else:
    #return 2 - 2*(x_cur)/x
    #return 1.7 - 1.6*(x_cur)/x
    return 1.4333 - 1.3333*(x_cur)/x

test_transform = transforms.Compose([
                    transforms.ToPILImage(), 
                    transforms.Resize((256,256)),
                    transforms.ToTensor(), 
                   #transforms.Normalize(0.5, 0.5)   #DONE
                ])

cdict = {0: 'c', 1: 'b', 2: 'g', 3: 'r', 4: 'c', 5: 'm', 6: 'k', 7: 'y', 8: 'w', 9: 'k'} #only 8 standard colors in plt

def pl_forward_LUS(test_input):    
    print(test_input.shape)
    if test_input.shape[0] == 1 and test_input.dim()==3:
      test_input = test_input.repeat_interleave(3,0)
    print(test_input.shape)
    for i in range(test_input.shape[1]):
        test_input[0,i,:] = test_input[0,i,:] * depth_decay_coeff(i, test_input.shape[1])
    test_input[1] = test_input[0]
    test_input_orig = test_input.clone().detach()
    test_input = (test_input[:-1]).repeat_interleave(5,0).unsqueeze(0)

    test_input_radon_bpm = test_input.clone().detach().to(args.device)
    test_input_radon_h = test_input_radon_bpm

    n=25
    theta_in = torch.linspace(0., 180., steps=180)
    theta_out = torch.linspace(90. - n, 90. + n, steps=90)
    r = Radon(test_input_radon_h[0][0].shape[1], theta_in, circle=False)
    ir = IRadon(test_input_radon_h[0][0].shape[1], theta_out, use_filter=HannFilter(), circle=False)
    sino_1 = r(test_input_radon_h[0][0].view(1,1,test_input_radon_h[0][0].shape[0],test_input_radon_h[0][0].shape[1]))
    n1 = 90 - n
    n2 = 90 + n
    test_input_radon_h[0][0] =  ir(sino_1[:,:,:, n1: n2])[0][0]

    #Applying vertical Radon filter
    test_input_radon_v = test_input_radon_bpm

    n = 30
    theta_in = torch.linspace(0., 180., steps=180)
    theta_out = torch.cat((torch.linspace(0,n, steps=45), torch.linspace(180 - n, 180, steps=45)))
    r = Radon(test_input_radon_v[0][1].shape[1], theta_in, circle = False)
    ir = IRadon(test_input_radon_v[0][1].shape[1], theta_out, use_filter=HannFilter(), circle=False)
    sino_1 = r(test_input_radon_v[0][1].view(1,1,test_input_radon_v[0][1].shape[0],test_input_radon_v[0][1].shape[1]))
    angles = torch.cat((sino_1[:,:,:,:n], sino_1[:,:,:,(180-n):]),dim=3)
    test_input_radon_v[0][1] =  ir(angles)[0][0]

    #Refining horizontal Radon features
    test_input_radon_bpm[0][0] = normalise(normalise(test_input_radon_bpm[0][0] ** 2) * test_input_orig[0])
    #Refining vertical Radon features
    test_input_radon_bpm[0][1] = normalise(normalise(test_input_radon_bpm[0][1]) * test_input_orig[0])

    # test_input_radon_bpm = test_input_radon_bpm.repeat_interleave(5,1)

    #Applying BPM on refined horizontal features extr. by Radon filter
    for ii in range(test_input_radon_bpm.shape[0]):
      for jj in range(0,5):
          test_input_radon_bpm[ii,jj] = torch.Tensor(bone_prob_map(test_input_radon_bpm[ii,jj].cpu().detach().numpy(), minwl = 3 + 3*jj)) * test_input_radon_bpm[ii,jj].cpu()
      #Applying BPM on refined vertical features extr. by Radon filter
    for ii in range(test_input_radon_bpm.shape[0]):  
      for jj in range(5,10):
          test_input_radon_bpm[ii,jj] = torch.Tensor(bone_prob_map(test_input_radon_bpm[ii,jj].cpu().detach().numpy(), minwl = 3 + 3*jj)) * test_input_radon_bpm[ii,jj].cpu()

    test_input_orig = test_input_orig.to('cpu')
    keypoints = gaussian_map(spatial_softmax(model_wghtKpt.model.point_net(test_input_radon_bpm)), std=model_wghtKpt.model.hlam_weights)
    keypoint_list = []
    #Since heatmap size is 32 in our case (instead of 16 used for 128x128 images)
    cur_fig = plt.figure()
    # plt.imshow(test_transform(image).permute(1,2,0), cmap='gray')
    print(test_input_orig.shape)
    plt.imshow(test_input_orig[0], cmap='gray')
    for j in range(10):  #since 10 heatmaps
        m = keypoints[0,j].view(1, -1).argmax(1)
        indices = torch.cat(((m / 64).view(-1, 1), (m % 64).view(-1, 1)), dim=1)
        x = indices[0,0].cpu().numpy()
        y = indices[0,1].cpu().numpy()
        keypoint_list.append([y*4, x*4])
        plt.scatter(y * 4, x * 4,
                    c=cdict[j], marker='*', cmap='RGB')
    print('returned')
    return

model_wghtKpt = plTransporter(get_config())
model_wghtKpt.to(args.device)
model_wghtKpt.load_state_dict(torch.load(args.ckpt)['state_dict'])



if args.img != '' and args.vid != '':
  print('Kindly enter the path to either a video or a Lung ultrasound image, not both')
  exit()

elif args.img != '':
  frame = plt.imread(args.img)
  print(frame.shape)
  if frame.ndim == 3:
    test_input = test_transform(frame[:,:,0]) 
  elif frame.ndim == 2:
    test_input = test_transform(frame) 
  pl_forward_LUS(test_input)
  print('recieved_img')
  print(os.path.join('output',args.img.split('/')[-1]))
  plt.savefig(os.path.join('output',args.img.split('/')[-1]))

elif args.vid != '' and args.frame_no != -1:
  try:
      import torchvision.io
      import av
  except:
      print('Kindly install torchvision and av')
      exit()

  num_sec, fps = torchvision.io.read_video_timestamps(args.vid, pts_unit = 'pts')
  num_sec = len(num_sec)
  frames, _, _ = torchvision.io.read_video(args.vid, pts_unit='pts')
  test_input = test_transform(frames[args.frame_no].permute(2,0,1))
  pl_forward_LUS(test_input)
  print('recieved_vid_frame')
  print(os.path.join('output','frame'+str(args.frame_no) + args.vid.split('/')[-1].split('.')[0] +'.jpg'))
  plt.savefig(os.path.join('output','frame'+str(args.frame_no) + args.vid.split('/')[-1].split('.')[0] +'.jpg'))

else:
  print('Kindly enter the path to either a video or a Lung ultrasound image')
  exit()