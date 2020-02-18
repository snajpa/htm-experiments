src = $(wildcard ./*.c)
obj = $(src:.c=.o)

LDFLAGS = -lm -fopenmp
CFLAGS = -Wall -O3 -ffast-math -mavx -march=native -fopt-info-vec-all -fopenmp

htm: $(obj)
	$(CC) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(obj) htm
