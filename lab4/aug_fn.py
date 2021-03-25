import albumentations as A

class Aug_fn():
  def __init__(self):
    pass
  
  def switch_func(self, name):
     return { 
       'randomBC': A.Compose([
         A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        ])
     }.get(name)
   
def main():
  aug = Aug_fn()
  kek = aug.switch_func("e")
  print(kek)

  
if __name__ == '__main__':
    main()
