from models.next_pandemics_gnn.NextPandemicsGNNModel import NextPandemicsGNNModel


class NextPandemicsGNNTrainer:
    def __init__(self, config):
        self.config = config
        self.model = NextPandemicsGNNModel(config)

    def train(self, log_callback=None):
        pass