def longest_common_subsequence(seq1, seq2):
    m = len(seq1)
    n = len(seq2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

def is_valid_sequence(seq, dataset):
    for data in dataset:
        if longest_common_subsequence(seq, data) >= 100:
            return False
    return True

def split_dataset(dataset):
    valid_data = []
    invalid_data = []

    for seq in dataset:
        if is_valid_sequence(seq, valid_data):
            valid_data.append(seq)
        else:
            invalid_data.append(seq)

    return valid_data, invalid_data
