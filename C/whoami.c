#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
	FILE *fp;
	char username[256];
	
	fp = popen("whoami", "r");
	if (fp == NULL) {
		fprintf(stderr, "Error executing whoami command.\n");
		exit(1);
	}

	if (fgets(username, sizeof(username), fp) != NULL) {
		printf("Username: %s", username);
	}
	
	pclose (fp);
	return 0;
}
