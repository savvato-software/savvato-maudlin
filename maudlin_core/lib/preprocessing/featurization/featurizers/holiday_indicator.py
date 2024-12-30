
def apply(data):
    # Example dictionary of holidays
    holidays = {
        '2023-01-01': 'New Year',
        '2023-01-03': 'Some Holiday',
        '2023-01-05': 'Another Holiday'
    }

    # Add a holiday feature to the summary dataframe
    data['holiday'] = data['date'].astype(str).apply(lambda x: 1 if x in holidays else 0)

    return data


