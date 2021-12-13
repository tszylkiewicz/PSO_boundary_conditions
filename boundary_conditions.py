import random


def absorbing(position, velocity, bounds):
    for i in range(len(position)):
        if position[i] < bounds[0]:
            position[i] = bounds[0]
            velocity[i] = 0
        if position[i] > bounds[1]:
            position[i] = bounds[1]
            velocity[i] = 0

    return position, velocity, False


def reflecting(position, velocity, bounds):
    for i in range(len(position)):
        if position[i] < bounds[0]:
            position[i] = bounds[0]
            velocity[i] *= -1.0
        if position[i] > bounds[1]:
            position[i] = bounds[1]
            velocity[i] *= -1.0

    return position, velocity, False


def dumping(position, velocity, bounds):
    for i in range(len(position)):
        if position[i] < bounds[0]:
            position[i] = bounds[0]
            velocity[i] *= -random.random()
        if position[i] > bounds[1]:
            position[i] = bounds[1]
            velocity[i] *= -random.random()

    return position, velocity, False


def invisible(position, velocity, bounds):
    skip = False
    for i in range(len(position)):
        if position[i] < bounds[0] or position[i] > bounds[1]:
            skip = True

    return position, velocity, skip


def invisible_reflecting(position, velocity, bounds):
    skip = False
    for i in range(len(position)):
        if position[i] < bounds[0] or position[i] > bounds[1]:
            velocity[i] *= -1.0
            skip = True

    return position, velocity, skip


def invisible_damping(position, velocity, bounds):
    skip = False
    for i in range(len(position)):
        if position[i] < bounds[0] or position[i] > bounds[1]:
            velocity[i] *= -random.random()
            skip = True

    return position, velocity, skip


def teleport(position, velocity, bounds):
    for i in range(len(position)):
        if position[i] < bounds[0]:
            diff = bounds[0] - position[i]
            position[i] = bounds[1] - diff
        if position[i] > bounds[1]:
            diff = position[i] - bounds[1]
            position[i] = bounds[0] + diff

    return position, velocity, False
