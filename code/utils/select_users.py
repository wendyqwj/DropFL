import numpy as np

_DIVIDE_RATIO = 0.4
_HIGH_PART_RATIO = 0.8


def health_select(health_dict: dict, select_num: int) -> list:
    """
    在健康值前_DIVIDE_RATIO的节点中随机选取_HIGH_PART_RATIO的参与方，
    剩下的节点中随机选取1-_HIGH_PART_RATIO的参与方
    """
    sorted_list = sorted(health_dict.items(), key=lambda x: x[1], reverse=True)
    print(f"sorted list: {sorted_list}")
    sorted_ids = [x[0] for x in sorted_list]
    high_part = int(select_num*_HIGH_PART_RATIO)
    low_part = select_num-high_part
    divide_index = int(len(health_dict)*_DIVIDE_RATIO)
    select_list = list(np.random.choice(
        sorted_ids[0:divide_index], high_part, replace=False))
    select_list += list(np.random.choice(
        sorted_ids[divide_index:], low_part, replace=False))
    return select_list


def random_select(health_dict: dict, select_num: int) -> list:
    return np.random.choice(list(health_dict.keys()), select_num, replace=False)
