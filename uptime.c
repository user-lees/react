#include <stdio.h>
#include <stdlib.h>

int main() {
	FILE *fp =popen("uptime", "r");
	if (fp == NULL) {
		fprintf(stderr, "Failed to execute command.\n");
		return 1;
	}

	char buffer[256];
	while (fgets(buffer, sizeof(buffer), fp) != NULL) {
		printf("%s", buffer);
	}

	pclose(fp);

	return 0;
}
