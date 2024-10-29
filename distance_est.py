def estimate_distance(box_height):
  
    
  h = 2 
  l = 500  
  distance = (h * l) / box_height
  return round(distance, 2)
