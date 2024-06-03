#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

int main(int argc, char *argv[]) {
	//입력된 디렉토리 생성
	for(int i = 1; i < argc; i++) {
		if (mkdir(argv[i], 0777) !=0) {
			perror("Error creating directory");
			exit(1);
		}
	}
	
	return 0;
}
