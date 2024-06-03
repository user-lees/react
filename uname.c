#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/utsname.h>
#include <string.h>

int main(int argc, char *argv[]) {
	int option;
	int print_all = 0;
	
	while ((option = getopt(argc, argv, "a")) != -1) {
		switch (option) {
			case 'a':
				print_all = 1;
				break;
			default:
				fprintf(stderr, "Invalid option: -%c\n", optopt);
				exit(1);
			}
		}
	struct utsname system_info;
	if (uname(&system_info) != 0) {
		fprintf(stderr, "Error retrieving system information.\n");
		exit(1);
	}

	if (print_all) {
		printf("System name: %s\n", system_info.sysname);
		printf("Node name: %s\n", system_info.nodename);
		printf("Release: %s\n", system_info.release);
		printf("Version: %s\n", system_info.version);
		printf("Machine: %s\n", system_info.machine);
	} else {
		printf("%s\n", system_info.sysname);
	}
	
	return 0;
}
