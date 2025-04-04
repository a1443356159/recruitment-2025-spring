
all:
	nvcc -O3 -Xcompiler "-O3 -march=native -fopenmp" -lcublas -lcudart -o winograd winograd.cc driver.cc

clean:
	rm -f winograd