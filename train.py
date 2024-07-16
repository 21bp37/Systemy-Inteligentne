from src import Physics
# import numpy as np
from trainer import SatelliteTrainer


def main():
    # radius = np.random.uniform(0.12, 0.38) * Physics.AU
    # target = 0.5 * Physics.AU + radius * np.array(
    #     [np.cos(theta := np.random.uniform(0, 2 * np.pi)), np.sin(theta)])

    target = [0.18 * Physics.AU, 0.294 * Physics.AU]
    trainer = SatelliteTrainer(target, num_episodes=600, size=128)
    trainer.train(render=True)  # , checkpoint='models/model_final.pth')
    # trainer.test('model/model.pth')


if __name__ == '__main__':
    main()
