#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    int operator_id;
    int tiling_id;
    int node_id_in_operator;
    int node_start_time;
    int processor_id_for_node;
} ScheduleNode;

void func() {
    srand(time(NULL));
    ScheduleNode ans[10];
    for (int i = 0; i < sizeof(ans) / sizeof(ans[0]); i++) {
        ans[i].operator_id = rand() % 11;
        ans[i].tiling_id = rand() % 11;
        ans[i].node_id_in_operator = rand() % 11;
        ans[i].node_start_time = rand() % 11;
        ans[i].processor_id_for_node = rand() % 11;
    }
    const int int_num_in_ScheduleNode = sizeof(ScheduleNode) / sizeof(int);
    const int num_of_ans = sizeof(ans) / sizeof(ans[0]);
    printf("[");
    for (int i = 0; i < num_of_ans; i++) {
        printf("[");
        int *p = (int *)(ans + i);
        for (int j = 0; j < int_num_in_ScheduleNode; j++) {
            printf("%d", p[j]);
            if (j < int_num_in_ScheduleNode - 1) {
                printf(", ");
            }
        }
        printf("]");
        if (i < num_of_ans - 1) {
        printf(", ");
        }
    }
    printf("]\n");
}