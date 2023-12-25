from abc import abstractclassmethod


class WrapperDataModule():
    def __init__(self) -> None:
        pass

    @abstractclassmethod
    def tain_dataloader(self):
        pass

    @abstractclassmethod
    def val_dataloader(self):
        pass

    @abstractclassmethod
    def test_dataloader(self):
        pass

    @abstractclassmethod
    def predict_dataloader(self):
        pass
