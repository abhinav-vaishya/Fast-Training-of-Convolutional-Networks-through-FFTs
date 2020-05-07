#pip install python-mnist
import numpy as np
from keras.datasets import mnist
import time

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def conv(image,n,k):
	h, w = image.shape
	if h != n:
		image = np.resize(image,(n,n))
	image = np.resize(image, (n+k-1, n+k-1))
	# output = np.zeros((n - k + 1, n - k + 1))
	output = np.zeros((n , n))
	filters = np.random.randn(k, k) / k*k
	# for i in range(n - k + 1):
	# 	for j in range(n - k + 1):
	# 		im_region = image[i:(i + k), j:(j + k)]
	# 		output[i, j] = np.sum(np.multiply(im_region,filters))
	for i in range(n):
		for j in range(n):
			im_region = image[i:(i + k), j:(j + k)]
			output[i, j] = np.sum(np.multiply(im_region,filters))
	
	# return output 

n = int(32)
k = int(7)
b = int(128)

# batch-size
f = open("tvsb_py.txt","w")
for i in range(15,151):
	print("batch size " + str(i))
	test_imgs  = test_images[:i]
	test_lbls  = test_labels[:i]
	tic = time.perf_counter()
	for im, label in zip(test_imgs,test_lbls):
		conv(im,n,k)
	toc = time.perf_counter()
	t = toc-tic
	s = str(i) + " " + str(t) +"\n"
	f.write(s)
f.close()

test_images  = test_images[:b]
test_labels  = test_labels[:b]

# input-size
f = open("tvsn_py.txt","w")
for i in range(8,81):
	print("input size " + str(i))
	tic = time.perf_counter()
	for im, label in zip(test_images,test_labels):
		conv(im,i,k)
	toc = time.perf_counter()
	t = toc-tic
	s = str(i) + " " + str(t) +"\n"
	f.write(s)
f.close()


# kernel-size
f = open("tvsk_py.txt","w")
for i in range(2,16):
	print("kernel size " + str(i))
	tic = time.perf_counter()
	for im, label in zip(test_images,test_labels):
		conv(im,n,i)
	toc = time.perf_counter()
	t = toc-tic
	s = str(i) + " " + str(t) +"\n"
	f.write(s)
f.close()









