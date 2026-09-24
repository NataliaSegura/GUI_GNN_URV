from data_pipeline.NextPandemicsGraphGenerator import NextPandemicsGraphGenerator
from models.next_pandemics_gnn.NextPandemicsGNNPredictor import NextPandemicsGNNPredictor
from models.next_pandemics_gnn.NextPandemicsGNNTrainer import NextPandemicsGNNTrainer


class NextPandemicsGNNFacade:

    def train(self, config, log_callback=None, progress_callback=None):
        trainer = NextPandemicsGNNTrainer(config)
        return trainer.train(log_callback=log_callback)

    def predict(self, config, log_callback=None, progress_callback=None):
        predictor = NextPandemicsGNNPredictor(config)
        return predictor.predict(log_callback=log_callback)

    def generate_data(self, config, log_callback=None, progress_callback=None):
        generator = NextPandemicsGraphGenerator(config)
        return generator.build_graphs(log_callback=log_callback)