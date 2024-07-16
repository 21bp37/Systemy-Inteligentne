from src import Physics
# import numpy as np
from trainer import SatelliteTrainer


def main():
    target = [0.18 * Physics.AU, 0.294 * Physics.AU]
    trainer = SatelliteTrainer(target, size=256)
    trainer.test('models/model_256.pth')


if __name__ == '__main__':
    main()
