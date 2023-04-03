def index2action_space(ind):
    i = ind // (3 * 1)
    j = (ind % (3 * 1)) // 1
    k = (ind % (3 * 1)) % 1
    return i, j, k


def action_index_to_position(action_index):
    i = action_index // (11 * 1)
    j = (action_index % (11 * 1)) // 1
    k = (action_index % (11 * 1)) % 1
    return i, j, k


def action_position_to_index(goal):
    return goal[0] * 11 + goal[1]
