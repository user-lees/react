#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
	if (argc <2) {
		fprintf(stderr, "Usage: %s <file1> [<file2> ...]\n", argv[0]);
		return 1;
	}
	
	//입력으로 받은 모든 파일 및 디렉토리 삭제
	for (int i = 1; i < argc; i++) {
		if (remove(argv[i] ) != 0) {
			perror("Error removing file or directory");
			return 1;
		}
	}

	return 0;
}
