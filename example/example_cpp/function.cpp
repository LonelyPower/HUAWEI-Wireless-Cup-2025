#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

class ScheduleNode {
public:
    // 成员变量
    int operator_id;
    int tiling_id;
    int node_id_in_operator;
    int node_start_time;
    int processor_id_for_node;

    // 构造函数
    ScheduleNode() {
        operator_id = 0;
        tiling_id = 0;
        node_id_in_operator = 0;
        node_start_time = 0;
        processor_id_for_node = 0;
    }

    // 随机初始化成员变量
    void randomize() {
        operator_id = rand() % 11;
        tiling_id = rand() % 11;
        node_id_in_operator = rand() % 11;
        node_start_time = rand() % 11;
        processor_id_for_node = rand() % 11;
    }

    // 打印对象
    void print() const {
        std::cout << operator_id << ", " << tiling_id << ", " << node_id_in_operator
                  << ", " << node_start_time << ", " << processor_id_for_node;
    }
};

void func() {
    srand(time(NULL));
    
    std::vector<ScheduleNode> ans(10);
    for (auto& node : ans) {
        node.randomize();
    }
    
    std::cout << "[";
    for (size_t i = 0; i < ans.size(); i++) {
        std::cout << "[";
        ans[i].print();
        std::cout << "]";
        if (i < ans.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}