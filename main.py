import argparse
from argparse import ArgumentError, ArgumentTypeError
from inference import Inference
from model import FashionModel
from train import Trainer
from data import TrainDataset

class ArgumentSelectError(Exception):
    pass


total_types = ['category', 'attribute1', 'attribute2', 'attribute3', 'attribute4', 'attribute5']

parser = argparse.ArgumentParser(
    prog='Fashion Category and Attribute Prediction',
    add_help=True,
    description='This program predicts categories, textures(attribute1),'
                'fabrics(attribute2), shapes(attribute3), parts(attribute4),'
                'and styles(attribute5).'
)

parser.add_argument(
    '-t', '--train',
    help='Trains model with `--train-data-dir` and `--train-data-csv`.',
    action='store_true'
)

parser.add_argument(
    '-p', '--predict',
    help='Inference model with `--sample-folder`.',
    action='store_true'
)

parser.add_argument(
    '--predict-type',
    help='Selects which type will be predicted. eg. `category`, `attribute1`.',
    type=str
)

parser.add_argument(
    '--train-type',
    help='Selects which type will be trained. eg. `category`, `attribute1`.',
    type=str
)

parser.add_argument(
    '--train-data-dir',
    help='Locate where is data folder.',
    type=str
)

parser.add_argument(
    '--log-dir',
    help='Locate where will training logs will be saved.',
    type=str
)

parser.add_argument(
    '--epoch',
    help='Number of epochs to train.',
    type=int
)

if __name__ == '__main__':
    args = parser.parse_args()

    try:
        if args.train:
            if args.train_data_dir is None:
                raise ArgumentSelectError('Train data directory not specified. Can not train!')
            elif args.log_dir is None:
                raise ArgumentSelectError('Log directory not specified. Can not train!')
            elif not any([args.train_type == train_type for train_type in total_types]):
                raise ArgumentSelectError('Train type not specified. Can not train!')
            else:
                print('Training!')
                train_dataset = TrainDataset(
                    image_dir=args.train_data_dir,
                    csv_path=f'data/dataset_csv/list_combined_{args.train_type}_small.tsv',
                    train_type=args.train_type,
                    batch_size=32,
                    shuffle=True,
                    random_seed=10
                )

                fm = FashionModel()
                fm.create_model(num_classes=train_dataset.num_classes)
                fm.model.summary()

                trainer = Trainer(
                    model=fm.model,
                    train_gen=train_dataset.train_generator,
                    val_gen=train_dataset.validation_generator,
                    epoch=args.epoch,
                )
                trainer.train(log_dir=args.log_dir)

                print('Training Finished!')
        elif args.predict:
            if not any([args.predict_type == pred_type for pred_type in total_types]):
                raise ArgumentSelectError('Predict type not specified. Can not predict.')
            else:
                print('Inference!')
                inf = Inference(model_path=f'models/{args.predict_type}.h5',
                                sample_dir='samples',
                                inference_type=args.predict_type,
                                inference_csv=f'data/{args.predict_type}.csv')

                inf.predict(save_result=True)
                print('Inference Completed!')

    except ArgumentSelectError as err:
        print(err)
        print('Please enter right arguments!')
