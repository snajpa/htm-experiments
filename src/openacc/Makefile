src = $(wildcard ./*.c)
obj = $(src:.c=.o)

CC=pgcc
LDFLAGS = -lm -mp -acc -ta=tesla:autocompare
CFLAGS = -lm -mp -acc -ta=tesla:autocompare -Minfo

htm: $(obj)
	$(CC) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(obj) htm
