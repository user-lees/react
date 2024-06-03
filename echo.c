#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>

int main(int argc, char *argv[]) {
    int option;
    int newline = 1;

    while ((option = getopt(argc, argv, "n")) != -1) {
        switch (option) {
            case 'n':
                newline = 1;
                break;
            case '?':
                fprintf(stderr, "Invalid option: -%c\n", optopt);
                exit(1);
        }
    }

    // Print the remaining command-line arguments
    for (int i = optind; i < argc; i++) {
        printf("%s ", argv[i]);
    }

    // Print a newline character if -n option is not provided
    if (!newline) {
        printf("\n");
    }

    return 0;
}
