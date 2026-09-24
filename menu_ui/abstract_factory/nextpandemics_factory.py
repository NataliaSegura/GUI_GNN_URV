from menu_ui.dialogs.nextpandemics.data_generation_dialog import DBGenerationDialog
from menu_ui.dialogs.nextpandemics.hyperparameter_dialog import HyperparameterSearchDialog
from menu_ui.dialogs.nextpandemics.predict_dialog import PredictDialog
from menu_ui.dialogs.nextpandemics.train_dialog import TrainDialog
from menu_ui.menu import Menu


def nextpandemics_menu(parent_window):
    return Menu(
        parent_window=parent_window,
        model_name="nextpandemics",
        db_dialog=DBGenerationDialog,
        train_dialog=TrainDialog,
        predict_dialog=PredictDialog,
        hp_dialog=HyperparameterSearchDialog,
        script_path="menu_ui/hyperparameter_adapter/nextpandemics_adapter.py",
    )
