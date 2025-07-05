import numpy as np
data = np.array([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High',   'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High',   'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High',   'Strong', 'Cool', 'Change', 'Yes']
])
def candidate_elimination(data):
    X = data[:, :-1]  # Features
    y = data[:, -1]   # Labels
    n_attr = X.shape[1]
    S = ['Ø'] * n_attr
    G = [['?' for _ in range(n_attr)]]
    print(f"\nInitial S: {S}")
    print(f"Initial G: {G}")
    for i in range(len(X)):
        instance, label = X[i], y[i]
        print(f"\n--- Example {i+1}: {instance}, Label: {label} ---")
        if label == 'Yes':
            for a in range(n_attr):
                if S[a] == 'Ø':
                    S[a] = instance[a]
                elif S[a] != instance[a]:
                    S[a] = '?'
            G = [g for g in G if all(g[a] == '?' or g[a] == instance[a] for a in range(n_attr))]

        elif label == 'No':
            G_new = []
            for g in G:
                for a in range(n_attr):
                    if g[a] == '?':
                        for val in np.unique(X[:, a]):
                            if val != instance[a]:
                                new_h = g.copy()
                                new_h[a] = val
                                if all(S[x] == '?' or new_h[x] == '?' or new_h[x] == S[x] for x in range(n_attr)):
                                    G_new.append(new_h)
            G = G_new
        print(f"S{i+1}: {S}")
        print(f"G{i+1}: {G}")
    return S, G
S_final, G_final = candidate_elimination(data)
print("\n=========================")
print("Final Specific Hypothesis (S):", S_final)
print("Final General Hypotheses (G):", G_final)
