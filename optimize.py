from utils.imgloader import load_data
import train as trainer

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


def run():
    n_trials = 10
    n_epochs = 20
    space = {
        'lr': hp.uniform('lr', 0.005, 0.05)
    }

    # load data
    train, val, num_to_name = load_data('data/101_ObjectCategories', 
                                        p_train=0.8, new_size=140)

    # define objective function
    def objective(params):
        score = trainer.run(train, val, len(num_to_name), 
            dim=128, num_epochs=n_epochs, opt=params)
        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()

    best = fmin(objective, space, 
                algo=tpe.suggest, 
                max_evals=n_trials, 
                trials=trials)

    print "best params:", best

if __name__ == '__main__':
    run()