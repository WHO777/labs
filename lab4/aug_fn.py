import albumentation as A

class aug_fn():
  
  def switch_func(name):
     return { 
       'randomBC': lambda name: A.Compose([
         A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        ])
     }.get(value)(name)
    
