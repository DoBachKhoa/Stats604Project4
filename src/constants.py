WEATHER_FEATURES = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wspd', 'pres']
WEATHER_FEATURES_HOURLY = ['temp', 'dwpt', 'rhum', 'prcp', 'wspd', 'pres']
WEATHER_FEATURES_HOURLY_COUNTS = { 'temp': 2, 'dwpt': 2, 'rhum': 1, 'prcp': 4, 'wspd': 1, 'pres': 1}
ZONES = ['AECO', 'AEPAPT', 'AEPIMP', 'AEPKPT', 'AEPOPT', 'AP', 'BC', 'CE', \
         'DAY', 'DEOK', 'DOM', 'DPLCO', 'DUQ', 'EASTON', 'EKPC', 'JC', 'ME', \
         'OE', 'OVEC', 'PAPWR', 'PE', 'PEPCO', 'PLCO', 'PN', 'PS', 'RECO', \
         'SMECO', 'UGI', 'VMEU']
MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
LEAPS = [2024, 2020, 2016]
HOURS = [f"H{int(h):02d}" for h in range(24)]
THANKSGIVING = {2016:24, 2017:23, 2018:22, 2019:28, 2020:26, 2021:25, 2022:24, 2023:23, 2024:28, 2025:27}
DAYLIGHT_START = {2016:'13', 2017:'12', 2018:'11', 2019:'10', 2020:'08', 2021:'14', 2022:'13', 2023:'12', 2024:'10', 2025:'09'}
DAYLIGHT_END = {2016:'06', 2017:'05', 2018:'04', 2019:'03', 2020:'01', 2021:'07', 2022:'06', 2023:'05', 2024:'03', 2025:'02'}
PRED_DAYS = [[3, -1], [4, -1], [5, -1], [6, 0], [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]
PRED_WEEK_START = 6
WEEKDAYS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
WEEKDAYS_DICT = {'mon':0, 'tue':1, 'wed':2, 'thu':3, 'fri':4, 'sat':5, 'sun':6}
