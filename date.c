#include <stdio.h>
#include <time.h>

int main() {
	// 현재 시간 구하기
	time_t t = time(NULL);
	struct tm*tm_info = localtime(&t);

	// 형식에 맞게 출력하기
	char buffer[80];
	strftime(buffer, sizeof(buffer), "%a %b %d %H:%M:%S %Y", tm_info);
	printf("%s\n", buffer);

	return 0;
}
