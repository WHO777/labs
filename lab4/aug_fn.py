import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Aug_fn():
  
  def get_method(self, name):
     return { 
       'randomBC': A.Compose([
         A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        ])
     }.get(name)
   
def main():
  aug = Aug_fn()
  kek = aug.get_method("randomBC")
  image = np.array(Image.open('0.jpg'), dtype=np.uint8)
  aug_image = kek(image=image)
  plt.imshow(aug_image['image'])
  plt.savefig('kek.jpg')

  
if __name__ == '__main__':
    main()
