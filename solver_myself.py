#!/usr/bin/env python3

import sys
import math
import random


from common import print_tour, read_input, format_tour


def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


# 経路の合計の長さを計算
def cal_path_length(cities, tour):
    N = len(cities)
    return sum(distance(cities[tour[i]], cities[tour[(i + 1) % N]]) for i in range(N))


# 貪欲法で初期解を作る
def greedy(cities, start_city):
    N = len(cities)

    dist = [[0] * N for i in range(N)]
    for i in range(N):
        for j in range(i, N):
            dist[i][j] = dist[j][i] = distance(cities[i], cities[j])

    unvisited_cities = set(range(0, N))
    unvisited_cities.remove(start_city)
    tour = [start_city]

    while unvisited_cities:
        next_city = min(unvisited_cities,
                        key=lambda city: dist[start_city][city])
        unvisited_cities.remove(next_city)
        tour.append(next_city)
        start_city = next_city
    return tour, dist


# 2-optで改良
def two_opt(tour, dist):
    N = len(tour)
    while True:
        cnt = 0
        for i in range(N - 2):
            for j in range(i + 2, N):
                l1 = dist[tour[i]][tour[i + 1]]
                l2 = dist[tour[j]][tour[(j + 1) % N]]
                l3 = dist[tour[i]][tour[j]]
                l4 = dist[tour[i + 1]][tour[(j + 1) % N]]
                if l1 + l2 > l3 + l4:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    cnt += 1
        if cnt == 0:
            break

    return tour


# Hinakoさんのコードを借りています
def expanded_distance_squared(city1, city2, expansion_rate):
    """
    Calculate the distance after expanding in the vertical direction
    expansion_rate: Expansion rate in the vertical direction
    """
    return (city1[0] - city2[0]) ** 2 + ((city1[1] - city2[1]) * expansion_rate) ** 2


def greedy_with_expansion(cities, expansion_rate, start_city):

    N = len(cities)

    dist = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i, N):
            dist[i][j] = dist[j][i] = expanded_distance_squared(
                cities[i], cities[j], expansion_rate)

    current_city = start_city
    unvisited_cities = set(range(0, N))
    unvisited_cities.remove(current_city)
    tour = [current_city]

    while unvisited_cities:
        next_city = min(unvisited_cities,
                        key=lambda city: dist[current_city][city])
        unvisited_cities.remove(next_city)
        tour.append(next_city)
        current_city = next_city
    return tour


def move_subsequence(tour, dist, subsequence_length):
    """
    Move subsequence E from between A and B to between C and D
    if the total distance becomes shorter
    dist: Array of distances between two cities
    subsequence_length: The length of E
    """
    N = len(tour)
    while True:
        count = 0  # Number of times tour was changed
        for a_index in range(N - 1):
            b_index = (a_index + subsequence_length + 1) % N
            e_indexes = [(a_index + 1 + i) % N for i in range(subsequence_length)]
            for c_index in range(N - 1):
                d_index = (c_index + 1) % N

                # When C,D are not included in e_indexes
                if d_index not in e_indexes and d_index != b_index:
                    A = tour[a_index]
                    B = tour[b_index]
                    C = tour[c_index]
                    D = tour[d_index]
                    E = [tour[index] for index in e_indexes]
                    if dist[A][E[0]] + dist[E[-1]][B] + dist[C][D] > dist[A][B] + dist[C][E[-1]] + dist[E[0]][D]:
                        # Replace the cities of E with -1. The indexes of C and D are not changed
                        for i in range(subsequence_length):
                            tour[(a_index + 1 + i) % N] = -1
                        # Reverse E and put E between C and D
                        for i in range(subsequence_length):
                            tour.insert(d_index, E[i])
                        # Remove the cities of E by removing -1
                        for i in range(subsequence_length):
                            tour = [city for city in tour if city != -1]
                        count += 1
        if count == 0:
            break
    return tour


# Kotoneさんのコードを借りました
def three_opt_swap(dist_matrix, path, i, j, k):
    """ swaps cities in a path if a shorter path connecting three edges can be found
    Args:
        dist_matrix: a NxN matrix (where N is the number of cities) dist_matrix[i][j] stores the distance between cities i and j
        path (list): a list of indices indicating the order of cities to travel, to be optimized
        i, j, k: indicies used to get a city in the path

    Returns:
        True if swap occured, otherwise False
    """
    p1, p2, p3, p4, p5, p6 = path[i-1], path[i], path[j-1], path[j], path[k-1], path[k % len(path)]
    d0 = dist_matrix[p1][p2] + dist_matrix[p3][p4] + dist_matrix[p5][p6]
    d1 = dist_matrix[p1][p4] + dist_matrix[p5][p3] + dist_matrix[p2][p6]
    d2 = dist_matrix[p1][p5] + dist_matrix[p4][p2] + dist_matrix[p3][p6]
    d3 = dist_matrix[p1][p3] + dist_matrix[p2][p5] + dist_matrix[p4][p6]
    d4 = dist_matrix[p1][p4] + dist_matrix[p5][p2] + dist_matrix[p3][p6]
    d5 = dist_matrix[p1][p3] + dist_matrix[p2][p4] + dist_matrix[p5][p6]
    d6 = dist_matrix[p1][p2] + dist_matrix[p3][p5] + dist_matrix[p4][p6]
    d7 = dist_matrix[p1][p5] + dist_matrix[p4][p3] + dist_matrix[p2][p6]

    if d1 < d0:
        path[i:k] = path[j:k] + path[j - 1:i - 1:-1]
        return True
    elif d2 < d0:
        path[i:k] = path[k - 1:j - 1:-1] + path[i:j]
        return True
    elif d3 < d0:
        path[i:k] = path[j - 1:i - 1:-1] + path[k - 1:j - 1:-1]
        return True
    elif d4 < d0:
        path[i:k] = path[j:k] + path[i:j]
        return True
    elif d5 < d0:
        path[i:j] = reversed(path[i:j])
        return True
    elif d6 < d0:
        path[j:k] = reversed(path[j:k])
        return True
    elif d7 < d0:
        path[i:k] = reversed(path[i:k])
        return True
    return False


def three_opt(path, dist_matrix):
    """ finds a path to travel using the three opt algorithm
    Args:
        path (list): a list of indices indicating the order of cities to travel, to be optimized
        dist_matrix: a NxN matrix (where N is the number of cities) dist_matrix[i][j] stores the distance between cities i and j
    Returns:
        path (list): a list of indices indicating the order of cities to travel found from the three opt algorithm
    """
    better = path.copy()
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 4):
            for j in range(i + 2, len(path) - 2):
                for k in range(j + 2, len(path)):
                    if three_opt_swap(dist_matrix, better, i, j, k):
                        improved = True
    return better


# 1-or-optで改良
def one_or_opt(tour, dist):
    N = len(tour)
    while True:
        cnt = 0
        for i in range(N):
            for j in range(N):
                if j != (i + 1) % N and (j + 1) % N != (i + 1) % N:
                    l1 = dist[tour[i]][tour[(i + 1) % N]]
                    l2 = dist[tour[(i + 1) % N]][tour[(i + 2) % N]]
                    l3 = dist[tour[i]][tour[(i + 2) % N]]
                    l4 = dist[tour[j]][tour[(j + 1) % N]]
                    l5 = dist[tour[j]][tour[(i + 1) % N]]
                    l6 = dist[tour[(j + 1) % N]][tour[(i + 1) % N]]
                    if l1 + l2 + l4 > l5 + l6 + l3:
                        change_node = tour.pop((i + 1) % N)
                        if (i + 1) % N < j:
                            tour.insert(j, change_node)
                        else:
                            tour.insert((j + 1) % N, change_node)
                        cnt += 1
        if cnt == 0:
            break

    return tour


# スタート地点を変えて探索
def change_start(cities):
    N = len(cities)
    min_tour = []
    min_path_length = 10**100
    print(N)
    if N == 2048:
        dist = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(i, N):
                dist[i][j] = dist[j][i] = distance(cities[i], cities[j])
        print("Distance array was created.")
        expansion_rate = 1.15
        temp_tour = greedy_with_expansion(cities, expansion_rate, 2)
        for _ in range(4):
            temp_tour = two_opt(temp_tour, dist)
            temp_tour = move_subsequence(temp_tour, dist, 9)
            temp_tour = move_subsequence(temp_tour, dist, 8)
            temp_tour = move_subsequence(temp_tour, dist, 7)
            temp_tour = move_subsequence(temp_tour, dist, 6)
            temp_tour = move_subsequence(temp_tour, dist, 5)
            temp_tour = move_subsequence(temp_tour, dist, 4)
            temp_tour = move_subsequence(temp_tour, dist, 3)
            temp_tour = move_subsequence(temp_tour, dist, 2)
            temp_tour = one_or_opt(temp_tour, dist)
            temp_path_length = cal_path_length(cities, temp_tour)
            if temp_path_length < min_path_length:
                min_tour = temp_tour
                min_path_length = temp_path_length
    elif N == 8192:
        dist = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(i, N):
                dist[i][j] = dist[j][i] = distance(cities[i], cities[j])
        print("Distance array was created.")
        expansion_rate = 1.15
        temp_tour = greedy_with_expansion(cities, expansion_rate, 0)
        for _ in range(4):
            temp_tour = two_opt(temp_tour, dist)
            temp_tour = three_opt(temp_tour, dist)
            temp_tour = move_subsequence(temp_tour, dist, 9)
            temp_tour = move_subsequence(temp_tour, dist, 8)
            temp_tour = move_subsequence(temp_tour, dist, 7)
            temp_tour = move_subsequence(temp_tour, dist, 6)
            temp_tour = move_subsequence(temp_tour, dist, 5)
            temp_tour = move_subsequence(temp_tour, dist, 4)
            temp_tour = move_subsequence(temp_tour, dist, 3)
            temp_tour = move_subsequence(temp_tour, dist, 2)
            temp_tour = one_or_opt(temp_tour, dist)
            temp_path_length = cal_path_length(cities, temp_tour)
            if temp_path_length < min_path_length:
                min_tour = temp_tour
                min_path_length = temp_path_length
    elif N == 5 or N == 8 or N == 16:
        for i in range(N):
            print(i)
            temp_tour, dist = greedy(cities, i)
            temp_tour = two_opt(temp_tour, dist)
            temp_tour = one_or_opt(temp_tour, dist)
            temp_path_length = cal_path_length(cities, temp_tour)
            if temp_path_length < min_path_length:
                min_tour = temp_tour
                min_path_length = temp_path_length
    elif N == 64:
        dist = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(i, N):
                dist[i][j] = dist[j][i] = distance(cities[i], cities[j])
        print("Distance array was created.")
        expansion_rate = 1.5
        temp_tour = greedy_with_expansion(cities, expansion_rate, 0)
        temp_tour = move_subsequence(temp_tour, dist, 5)
        for _ in range(2):
            temp_tour = two_opt(temp_tour, dist)
            temp_tour = three_opt(temp_tour, dist)
            temp_tour = move_subsequence(temp_tour, dist, 4)
            temp_tour = move_subsequence(temp_tour, dist, 3)
            temp_tour = move_subsequence(temp_tour, dist, 2)
            temp_tour = one_or_opt(temp_tour, dist)
            temp_path_length = cal_path_length(cities, temp_tour)
            if temp_path_length < min_path_length:
                min_tour = temp_tour
                min_path_length = temp_path_length
    elif N == 128:
        dist = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(i, N):
                dist[i][j] = dist[j][i] = distance(cities[i], cities[j])
        print("Distance array was created.")
        expansion_rate = 1.15
        for i in range(N):
            temp_tour = greedy_with_expansion(cities, expansion_rate, i)
            temp_tour = move_subsequence(temp_tour, dist, 6)
            temp_tour = move_subsequence(temp_tour, dist, 5)
            for _ in range(2):
                temp_tour = two_opt(temp_tour, dist)
                temp_tour = move_subsequence(temp_tour, dist, 4)
                temp_tour = move_subsequence(temp_tour, dist, 3)
                temp_tour = move_subsequence(temp_tour, dist, 2)
                temp_tour = one_or_opt(temp_tour, dist)
                temp_path_length = cal_path_length(cities, temp_tour)
                if temp_path_length < min_path_length:
                    min_tour = temp_tour
                    min_path_length = temp_path_length
    else:  # N == 512
        dist = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(i, N):
                dist[i][j] = dist[j][i] = distance(cities[i], cities[j])
        print("Distance array was created.")
        expansion_rate = 1.37
        temp_tour = greedy_with_expansion(cities, expansion_rate, 0)
        temp_tour = move_subsequence(temp_tour, dist, 6)
        temp_tour = move_subsequence(temp_tour, dist, 5)
        for _ in range(2):
            temp_tour = two_opt(temp_tour, dist)
            temp_tour = move_subsequence(temp_tour, dist, 4)
            temp_tour = move_subsequence(temp_tour, dist, 3)
            temp_tour = move_subsequence(temp_tour, dist, 2)
            temp_tour = one_or_opt(temp_tour, dist)
            temp_path_length = cal_path_length(cities, temp_tour)
            if temp_path_length < min_path_length:
                min_tour = temp_tour
                min_path_length = temp_path_length

    return min_tour


if __name__ == '__main__':
    assert len(sys.argv) > 1
    cities = read_input(sys.argv[1])
    tour = change_start(cities)
    print_tour(tour)

    with open(sys.argv[2], mode='w') as f:
        f.write(format_tour(tour))
