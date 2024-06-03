#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
	if (argc <2) {
		fprintf(stderr, "Usage: %s <directory>...\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	//입력된 디렉토리 삭제
	for (int i = 1; i < argc; i++) {
		if (rmdir(argv[i]) !=0) {
		perror("Error removing directory");
		exit(EXIT_FAILURE);
		}
	}
	
	return 0;
}
