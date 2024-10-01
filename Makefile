build:
	docker build --tag donorsearch .

run:
	docker run --rm -it -p 8000:8000 \
		-v  /mnt/c/Users/dmitr/VSC_holder/DonorSearch/transformed_images:/app/transformed_images \
		-v  /mnt/c/Users/dmitr/VSC_holder/DonorSearch/input_images:/app/input_images \
		donorsearch
  
      

