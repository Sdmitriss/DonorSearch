build:
      
	    docker build --tag donorsearch

run:
        docker run -rm -it -p 8000:8000\
		-v ./transform_images:app/transform_images\
		-v ./input_images:app/input_images\
		donorsearch
  
      

