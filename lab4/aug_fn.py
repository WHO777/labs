import albumentation as A

class Aug_fn():
  
  def switch_func(name):
     return { 
       'randomBC': lambda name: A.Compose([
         A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        ])
     }.get(value)(name)
    
def main():
  aug = Aug_fn()
  kek = aug.switch_func('randomBC')
  print(kek)
