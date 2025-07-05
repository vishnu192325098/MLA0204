import pandas as pd
def find_s_algorithm(data):
    hypothesis = None
    for index, row in data.iterrows():
        if row['Target'] == 'Yes':
            instance = row[:-1].values
            if hypothesis is None:
                hypothesis = instance.copy()
            else:
                for i in range(len(hypothesis)):
                    if hypothesis[i] != instance[i]:
                        hypothesis[i] = '?'
    return hypothesis
data = pd.DataFrame([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
], columns=['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'Target'])
final_hypothesis = find_s_algorithm(data)
print("Final Hypothesis:", final_hypothesis)
