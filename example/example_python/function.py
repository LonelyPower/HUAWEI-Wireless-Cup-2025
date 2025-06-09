import random
from typing import List, Tuple


def func():
    ans: List[List[int]] = []
    schedule_node: Tuple[int, int, int, int, int]
    for i in range(10):
        operator_id: int = random.randint(0, 10)
        tiling_id: int = random.randint(0, 10)
        node_id_in_operator: int = random.randint(0, 10)
        node_start_time: int = random.randint(0, 10)
        processor_id_for_node: int = random.randint(0, 10)
        schedule_node: Tuple[int, int, int, int, int] = (operator_id, tiling_id, node_id_in_operator, node_start_time, processor_id_for_node)

        ans.append(list(schedule_node))
    print(ans)

if __name__ == '__main__':
    func()