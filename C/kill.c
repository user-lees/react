#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

int main(int argc, char* argv[]) {
            if (argc != 3) {
                            printf("Usage: %s <signal_number> <pid>\n", argv[0]);
                                    exit(EXIT_FAILURE);
                                        }

                int signal_num = atoi(argv[1]);
                    pid_t pid = atoi(argv[2]);

                        if (kill(pid, signal_num) == -1) {
                                        perror("kill");
                                                exit(EXIT_FAILURE);
                                                    }

                            printf("Signal %d sent to process %d\n", signal_num, pid);

                                return 0;
}
                
