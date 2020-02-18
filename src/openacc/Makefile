src = $(wildcard ./*.c)
obj = $(src:.c=.o)

CC=pgcc
LDFLAGS = -lm -mp
CFLAGS = -lm -O3 -mp

htm: $(obj)
	$(CC) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(obj) htm
