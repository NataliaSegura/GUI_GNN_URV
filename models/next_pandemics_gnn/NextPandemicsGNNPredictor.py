from models.next_pandemics_gnn.NextPandemicsGNNModel import NextPandemicsGNNModel


class NextPandemicsGNNPredictor:
    def __init__(self, config):
        self.config = config
        self.model = NextPandemicsGNNModel(config)

    def predict(self, log_callback=None):
        pass