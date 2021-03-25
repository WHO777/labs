import albumentations as A
import numpy as np
import matplotlib.pyplot as plt

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
  image = kek(image=np.zeros((224,224,1), dtype=np.uint8))
  print(image)
  plt.imshow(image)
  plt.savefig('kek.jpg')
  
  print(kek)

  
if __name__ == '__main__':
    main()
