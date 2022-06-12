import random
import cv2
import numpy as np
import albumentations as albu
from numpy.random import choice
import skimage.exposure


def hair_mask(hairs,IMAGE_SIZE):
           
    mask_to_chose = choice(np.arange(14), 1,p=[0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.075,0.075,0.075,0.075])[0]
    mask = hairs[mask_to_chose]
    hair_trans = albu.Compose([albu.ShiftScaleRotate(rotate_limit=[-45,45],scale_limit=[-0.1,0.1], shift_limit=[-0.1,0.15],border_mode=3,p=1.)])
    mask = hair_trans(image = mask)['image']
    mask = cv2.resize(mask/255,(IMAGE_SIZE,IMAGE_SIZE),cv2.INTER_CUBIC)
    mask[mask == 1.] =  255
    mask[mask != 255.] = 0

    return mask


def mm_aug(image,GT,mm_array):
  colormap = np.array([[104,43,159],[191,64,191],[106,51,170],[211,63,93],[99,41,112]])
  desired_color = colormap[random.randint(0,4)]

  mask = mm_array[random.randint(0,13)].astype('uint8')
  aug_transformer = albu.Compose([albu.HorizontalFlip(p=0.5),albu.VerticalFlip(p=0.5),albu.ShiftScaleRotate(rotate_limit=[-45,45],scale_limit=[-0.1,0.1], shift_limit=[-0.1,0.15],border_mode=3,p=1.)])
  mask = aug_transformer(image = mask)['image']

  final_mask = cv2.bitwise_and(GT,GT, mask = mask)
  
  swatch = np.full((256,256,3), desired_color, dtype=np.uint8)

  facemask = final_mask

  ave_color = cv2.mean(image,mask=facemask)[:3]

  diff_color = desired_color - ave_color
  diff_color = np.full_like(image, diff_color, dtype=np.uint8)

  # shift input image color
  new_img = image+diff_color

  # antialias mask, convert to float in range 0 to 1 and make 3-channels
  facemask = cv2.GaussianBlur(facemask, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
  facemask = skimage.exposure.rescale_intensity(facemask, in_range=(100,150), out_range=(0,1)).astype(np.float32)
  facemask = cv2.merge([facemask,facemask,facemask])

  # combine img and new_img using mask
  result = image * (1-facemask) + new_img * (facemask)
  result = result.clip(0,255).astype(np.uint8)
  return result



def ruler_aug(image,GT,ruler_array):
  mask = ruler_array[random.randint(0,13)].astype('uint8')
  aug_transformer = albu.Compose([albu.HorizontalFlip(p=0.5),albu.VerticalFlip(p=0.5),albu.ShiftScaleRotate(rotate_limit=[-45,45],scale_limit=[-0.1,0.1], shift_limit=[-0.1,0.15],border_mode=3,p=1.)])
  aug_mask = aug_transformer(image = mask)['image'] 
  aug_mask = cv2.bitwise_not(aug_mask)
  update_mask = cv2.bitwise_not(cv2.bitwise_and(aug_mask,aug_mask,mask=GT[:,:,0]))
  augmented_image = cv2.bitwise_and(image, update_mask, mask=update_mask[:,:,0])
  return augmented_image