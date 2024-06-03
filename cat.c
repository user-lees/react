#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void cat(const char* filename, int showLineNumbers, int squeezeEmptyLines) {
    FILE* fp = fopen(filename, "r");

    if (fp == NULL) {
        printf("Failed to open file.\n");
        return;
    }

    int lineNum = 1;
    int prevEmptyLine = 0;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), fp)) {
        if (squeezeEmptyLines && buffer[0] == '\n') {
            if (prevEmptyLine)
                continue;
            else
                prevEmptyLine = 1;
        } else {
            prevEmptyLine = 0;
            if (showLineNumbers)
                printf("%d: ", lineNum++);
            printf("%s", buffer);
        }
    }

    fclose(fp);
}

int main(int argc, char* argv[]) {
    int opt;
    int showLineNumbers = 0;
    int squeezeEmptyLines = 0;

    while ((opt = getopt(argc, argv, "nbs")) != -1) {
        switch (opt) {
            case 'n':
                showLineNumbers = 1;
                break;
            case 'b':
                showLineNumbers = 1;
                squeezeEmptyLines = 1;
                break;
            case 's':
                squeezeEmptyLines = 1;
                break;
            default:
                fprintf(stderr, "Usage: cat [-n] [-b] [-s] file\n");
                return 1;
        }
    }

    if (optind >= argc) {
        fprintf(stderr, "Usage: cat [-n] [-b] [-s] file\n");
        return 1;
    }

    const char* filename = argv[optind];
    cat(filename, showLineNumbers, squeezeEmptyLines);

    return 0;
}
