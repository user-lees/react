#include <stdio.h>
#include <string.h>

void reverse(char *str) {
	int start = 0;
	int end = strlen(str) -1;

	while (start < end) {
		//문자 위치를 스왑합니다.
		char temp = str[start];
		str[start] = str[end];
		str[end] = temp;

		//다음 문자로 이동합니다.
		start++;
		end--;
	}
}

int main(int argc, char *argv[]) {
	if (argc !=2) {
		printf("Usage: %s <string>\n", argv[0]);
		return 1;
	}

	char *input = argv[1];

	//입력된 문자열을 뒤집습니다.
	reverse(input);

	//뒤집힌 문자열을 출력합니다.
	printf("%s\n", input);

	return 0;
}
