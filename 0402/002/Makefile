CC = gcc
CFLAGS = -Wall
SOURCES = hello.c

all: hello
	OBJECTS = $(SOURCES:.c=.o)

hello: $(OBJECTS)
	$(CC) $(CFLAGS) -o hello $(OBJECTS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f hello $(OBJECTS)
