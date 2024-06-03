#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <getopt.h>

int main(int argc, char* argv[]) {
    int option;
    int signal_number = SIGTERM; // 기본 시그널 번호: SIGTERM
    int list_signals = 0;

    while ((option = getopt(argc, argv, "9l")) != -1) {
        switch (option) {
            case '9':
                signal_number = SIGKILL;
                break;
            case 'l':
                list_signals = 1;
                break;
            default:
                fprintf(stderr, "Usage: %s [-9 PID | -l]\n", argv[0]);
                return -1;
        }
    }

    if (list_signals) {
        printf("Available signals:\n");
        for (int i = 1; i < NSIG; i++) {
            if (i == SIGKILL || i == SIGSTOP) {
                continue; // SIGKILL과 SIGSTOP은 명령어로 사용할 수 없음
            }
            printf("%d: %s\n", i, sys_siglist[i]);
        }
    } else {
        if (optind >= argc) {
            fprintf(stderr, "Usage: %s [-9 PID | -l]\n", argv[0]);
            return -1;
        }

        pid_t pid = atoi(argv[optind]);
        if (kill(pid, signal_number) == -1) {
            perror("kill");
            return -1;
        }
    }

    return 0;
}
