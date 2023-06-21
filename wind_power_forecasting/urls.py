""" Url and Paths for the wind power forecasting project.

This module contains the url and paths for the wind power forecasting project.
"""

__authors__ = ["Anthony Christoforou", "Ethan Arm"]
__contact__ = ["anthony.christoforou@etu.unige.ch", "ethan.arm@etu.unige.ch"]
__copyright__ = "Copyright 2023, Unige"
__credits__ = ["Anthony Christoforou", "Ethan Arm"]
__date__ = "2023/05/06"
__deprecated__ = False
__license__ = "GPLv3"
__maintainer__ = "Anthony Christoforou"
__status__ = "Development"
__version__ = "0.0.1"

data_url = "https://bj.bcebos.com/v1/ai-studio-online/85b5cb4eea5a4f259766f42a448e2c04a7499c43e1ae4cc28fbdee8e087e2385?responseContentDisposition=attachment%3B%20filename%3Dwtbdata_245days.csv&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-05-05T14%3A17%3A03Z%2F-1%2F%2F5932bfb6aa3af1bcfb467bf2a4a6877f8823fe96c6f4fd0d4a3caa722354e3ac"
relative_position_url = "https://bj.bcebos.com/v1/ai-studio-online/e927ce742c884955bf2a667929d36b2ef41c572cd6e245fa86257ecc2f7be7bc?responseContentDisposition=attachment%3B%20filename%3Dsdwpf_baidukddcup2022_turb_location.CSV&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-04-11T08%3A27%3A09Z%2F-1%2F%2Fcf377452dbd186873680f2f0fe39200b3de86083a036da220ab8a02abc5a8032"
    
data_dir = "./wind_power_forecasting/data/wtbdata_245days.csv"
relative_position_file = "./wind_power_forecasting/data/sdwpf_baidukddcup2022_turb_location.CSV"