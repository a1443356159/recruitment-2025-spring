CFLAG = -O3

all:
	nvcc -O3 -Xcompiler "-O3 -march=native -fopenmp" -lcublas -o winograd winograd.cc driver.cc

clean:
	rm -f winograd