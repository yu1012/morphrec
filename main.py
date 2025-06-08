""" Main file for training/evaluating the model """
import os
import sys

from configs.config import parse
from init import init_env, init_train, init_test
from train_fn import train_and_evaluate
from eval_fn import evaluate

if __name__ == '__main__':
    cfg = parse(sys.argv[1])
    cfg = init_env(cfg)

    model, train_dl, val_dl, optimizer, scheduler, loss_fn, metric, logger, log_path = init_train(cfg)
    train_and_evaluate(cfg, model, train_dl, val_dl, optimizer, scheduler,
                        loss_fn, metric, logger, log_path) 
    print("Training Done")
    
    model, test_dl, metric, loss_fn, log_path = init_test(cfg)
    _, _, metrics_string, _ = evaluate(model, loss_fn, test_dl, metric, cfg, True, log_path, mode="test", save_pkl=True)
    print("Evaluation Result")
    print(metrics_string)
    with open(os.path.join(log_path, 'eval_results.txt'), 'w') as f:
        f.write(metrics_string)
        f.close()
