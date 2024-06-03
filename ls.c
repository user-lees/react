#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <getopt.h>
#include <string.h>

void print_file_info(const char* filename) {
    struct stat file_info;
    if (stat(filename, &file_info) == -1) {
        perror("stat");
        return;
    }

    printf("File: %s\n", filename);
    printf("Size: %lld bytes\n", (long long)file_info.st_size);
    printf("Permissions: %o\n", file_info.st_mode & 0777);
    printf("\n");
}

void list_files(int show_hidden, int show_almost_all, int show_details) {
    DIR* dir;
    struct dirent* entry;

    dir = opendir(".");
    if (dir == NULL) {
        perror("opendir");
        return;
    }

    while ((entry = readdir(dir)) != NULL) {
        if (!show_hidden && !show_almost_all && entry->d_name[0] == '.') {
            continue;
        }

        if (show_almost_all && (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)) {
            continue;
        }

        if (show_details) {
            print_file_info(entry->d_name);
        } else {
            printf("%s\n", entry->d_name);
        }
    }

    closedir(dir);
}

int main(int argc, char* argv[]) {
    int show_hidden = 0;
    int show_almost_all = 0;
    int show_details = 0;
    int opt;
    int option_index = 0;

    static struct option long_options[] = {
        {"all", no_argument, 0, 'a'},
        {"almost-all", no_argument, 0, 'A'},
        {"long", no_argument, 0, 'l'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "aAl", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'a':
                show_hidden = 1;
                break;
            case 'A':
                show_almost_all = 1;
                break;
            case 'l':
                show_details = 1;
                break;
            default:
                fprintf(stderr, "Usage: %s [-a] [-A] [-l] [--all] [--almost-all] [--long]\n", argv[0]);
                return -1;
        }
    }

    list_files(show_hidden, show_almost_all, show_details);

    return 0;
}

