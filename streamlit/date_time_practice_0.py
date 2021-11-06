from datetime import datetime, timedelta

#  ('1W', '1M', '6M', '1Y', '5Y')


days = input('days: ')


def select_day(selection):
    if selection == '1W':
        output = 7
    elif selection == '1M':
        output = 31
    elif selection == '6M':
        output = 186
    elif selection == '1Y':
        output = 365
    elif selection == '5Y':
        output = 1825
    else:
        output = 365
    return output


time_range = select_day(days)
print(time_range)

FORMAT = '%Y-%m-%d'
end = datetime.now().strftime(FORMAT)

print('end date: ', end)
end = datetime.strptime(end, '%Y-%m-%d')
time_window = timedelta(days=time_range)
start_time = end - time_window
start_time = start_time.strftime(FORMAT)
print('start time: ', start_time)
# st.write('Start time:', start_time)
