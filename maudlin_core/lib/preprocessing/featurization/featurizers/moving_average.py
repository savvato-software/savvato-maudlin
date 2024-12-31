
def apply(data, periods, data_field_name, include_differences=False):

    # Calculate moving averages
    for period in periods:
        data[f"MA_{period}"] = data[data_field_name].rolling(window=period).mean()

    # Calculate differences between moving averages if the flag is set
    if include_differences and len(periods) > 1:
        for i in range(len(periods)):
            for j in range(i + 1, len(periods)):
                period_1 = periods[i]
                period_2 = periods[j]
                col_name = f"MA_diff_{period_1}_{period_2}"
                data[col_name] = data[f"MA_{period_2}"] - data[f"MA_{period_1}"]

    return data


