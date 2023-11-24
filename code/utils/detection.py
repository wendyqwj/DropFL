def detect(att_users: list, health_ids: list) -> float:
    assert len(health_ids) >= len(att_users)
    count = 0
    for id in health_ids[:len(att_users)]:
        if id in att_users:
            count += 1
    return count/len(att_users)


def detratio_vs_num(sorted_ids: list, att_users: list, step=10) -> list:
    ratio_list = list()
    att_users = set(att_users)
    for end_index in range(step, len(sorted_ids)+1, step):
        selecet_ids = set(sorted_ids[:end_index])
        intersection = selecet_ids & att_users
        ratio_list.append(len(intersection)/len(att_users))
    return ratio_list
