import pytest
from wind_power_forecasting.data import WindDataset

class TestWindDataset:
    data_dir = "./wind_power_forecasting/data/wtbdata_245days.csv"
    relative_position_file = "./wind_power_forecasting/data/sdwpf_baidukddcup2022_turb_location.CSV"
    dataset = WindDataset(data_dir, relative_position_file)

    def test_size(self):
        assert len(self.dataset) == 4678002
