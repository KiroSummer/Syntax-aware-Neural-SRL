def color(m, n):  #
    if n == 1:
        return m
    elif n == 2:
        return m * (m - 1)
    elif n == 3:
        return m * (m - 1) * (m - 2)
    return m * pow(m - 1, n - 1) - color(m, n - 1)


if __name__ == "__main__":
    print color(5, 5)
